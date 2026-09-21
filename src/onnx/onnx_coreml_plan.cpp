#include "onnx_coreml_plan.h"

#include <algorithm>
#include <cstdlib>
#include <map>
#include <vector>

namespace
{

// The line's tail after `marker`, or npos.
size_t after(const std::string &line, const char *marker, size_t from = 0)
{
  const size_t at = line.find(marker, from);
  if (at == std::string::npos)
    return at;
  return at + std::string(marker).size();
}

// "ios17.matmul" -> "matmul": the version prefix says nothing a reader needs.
std::string bareOp(const std::string &op)
{
  const size_t dot = op.find('.');
  return dot == std::string::npos ? op : op.substr(dot + 1);
}

// The friendly name of an MLComputeDevice class, for the reason.
std::string unitNameOf(const std::string &cls)
{
  if (cls == "MLNeuralEngineComputeDevice")
    return "the Neural Engine";
  if (cls == "MLGPUComputeDevice")
    return "the GPU";
  if (cls == "MLCPUComputeDevice")
    return "the CPU";
  return cls.empty() ? "no unit" : cls;
}

} // namespace

OnnxCoremlPlan onnxParseCoremlPlan(const std::string &text)
{
  OnnxCoremlPlan plan;
  size_t pos = 0;
  while (pos < text.size())
  {
    size_t nl = text.find('\n', pos);
    if (nl == std::string::npos)
      nl = text.size();
    const std::string line = text.substr(pos, nl - pos);
    pos = nl + 1;

    // NSLog prefixes each line with a timestamp and "process[pid:tid]";
    // the markers are searched for, not anchored.
    if (size_t e = after(line, "Error loading compute plan: "); e != std::string::npos)
    {
      if (plan.failure.empty())
        plan.failure = line.substr(e);
      continue;
    }
    if (line.find("profile function : timeout") != std::string::npos)
    {
      if (plan.failure.empty())
        plan.failure = "Core ML did not answer for the compute plan within the provider's five-minute wait";
      continue;
    }
    if (line.find("Error loading program from compute plan") != std::string::npos)
    {
      if (plan.failure.empty())
        plan.failure = "the compiled model is not an ML Program, so it has no compute plan";
      continue;
    }

    const size_t opAt = after(line, "Operation: ");
    if (opAt == std::string::npos)
      continue;
    const size_t opEnd = line.find(", Device Usage: ", opAt);
    if (opEnd == std::string::npos)
      continue;
    OnnxCoremlPlanOp op;
    op.op = line.substr(opAt, opEnd - opAt);

    // "<MLCPUComputeDevice: 0x103a913f0>" -- or "(null)" for an operation
    // the plan places nowhere, which the native backend skips too.
    size_t devAt = opEnd + std::string(", Device Usage: ").size();
    const size_t costAt = line.find(", Estimated Cost: ", devAt);
    std::string dev = line.substr(devAt, costAt == std::string::npos ? std::string::npos
                                                                     : costAt - devAt);
    if (!dev.empty() && dev[0] == '<')
      dev.erase(0, 1);
    if (const size_t colon = dev.find(':'); colon != std::string::npos)
      dev.resize(colon);
    while (!dev.empty() && (dev.back() == '>' || dev.back() == ' '))
      dev.pop_back();
    if (dev.empty() || dev == "(null)")
      continue;
    op.device = dev;

    if (costAt != std::string::npos)
      op.cost = std::strtod(line.c_str() + costAt + std::string(", Estimated Cost: ").size(),
                            nullptr);
    if (op.cost < 0.0)
      op.cost = 0.0;
    plan.known = true;
    plan.ops.push_back(std::move(op));
  }
  return plan;
}

std::string onnxCoremlPlanOffDeviceReason(const OnnxCoremlPlan &plan,
                                          const std::string &unitClass,
                                          const std::string &unitName)
{
  // The same rule as the native backend's onDevice(): the cost carried by
  // operations placed elsewhere, against a 5% allowance for glue.
  constexpr double kOffDeviceShare = 0.05;

  if (!plan.known || unitClass.empty())
    return std::string();

  double total = 0.0, off = 0.0;
  size_t offOps = 0;
  for (const auto &op : plan.ops)
  {
    total += op.cost;
    if (op.device != unitClass)
    {
      off += op.cost;
      offOps++;
    }
  }
  if (offOps == 0)
    return std::string();

  // A plan that costs nothing anywhere has said where the work goes and
  // nothing about how much of it: count operations instead of weighing
  // them.  Weighing would pass a session whose every operation moved.
  const bool weightless = total <= 0.0;
  const double share = weightless ? (double)offOps / (double)plan.ops.size()
                                  : off / total;
  if (share <= kOffDeviceShare)
    return std::string();

  // Name what moved, where it went, and how much of the work it was: the
  // significant operations first, by cost, without repeating a type.
  std::map<std::string, double> moved;
  std::map<std::string, size_t> where;
  for (const auto &op : plan.ops)
    if (op.device != unitClass)
    {
      moved[bareOp(op.op)] += weightless ? 1.0 : op.cost;
      where[op.device]++;
    }
  std::string ops;
  {
    // Largest share first; a plan of two dozen operation types would
    // otherwise read as a list of reshapes.
    std::vector<std::pair<double, std::string>> byCost;
    for (const auto &kv : moved)
      byCost.emplace_back(kv.second, kv.first);
    std::sort(byCost.begin(), byCost.end(),
              [](const auto &a, const auto &b) { return a.first > b.first; });
    size_t n = 0;
    for (const auto &e : byCost)
    {
      if (n++ == 4)
      {
        ops += ", ...";
        break;
      }
      if (!ops.empty())
        ops += ", ";
      ops += e.second;
    }
  }
  std::string dest;
  {
    // Where they went -- almost always one place.
    std::string best;
    size_t bestN = 0;
    for (const auto &kv : where)
      if (kv.second > bestN)
      {
        bestN = kv.second;
        best = kv.first;
      }
    dest = unitNameOf(best);
    if (where.size() > 1)
      dest += " and elsewhere";
  }
  const int pct = (int)(share * 100.0 + 0.5);
  return "Core ML's compute plan sends " + ops + " (" + std::to_string(pct) + "% of " +
         (weightless ? "the operations" : "the estimated cost") + ") to " + dest +
         " rather than " + unitName + ", so this would not be " +
         (unitName.compare(0, 4, "the ") == 0 ? "a " + unitName.substr(4) : unitName) +
         " number";
}

OnnxCoremlUnit onnxCoremlUnitFor(const std::string &computeUnits)
{
  if (computeUnits == "CPUAndNeuralEngine" || computeUnits == "ALL")
    return {"MLNeuralEngineComputeDevice", "the Neural Engine"};
  if (computeUnits == "CPUAndGPU")
    return {"MLGPUComputeDevice", "the GPU"};
  return {};
}
