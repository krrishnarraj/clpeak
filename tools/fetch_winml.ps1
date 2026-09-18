# fetch_winml.ps1 -- stage the Windows ML runtime for clpeak's --onnx-winml.
#
#   tools\fetch_winml.ps1 [-Version 2.3.42] [-Arch x64|arm64] [-Dest <dir>]
#
# Windows ML (Windows 11 24H2+) installs the vendor execution providers --
# Qualcomm QNN, Intel OpenVINO, AMD Vitis AI, NVIDIA TensorRT for RTX -- from
# the Microsoft Store and hands an app the path of each one's plugin library.
# The API for that lives in Microsoft.Windows.AI.MachineLearning.dll, which
# Microsoft ships in the Microsoft.Windows.AI.MachineLearning NuGet package
# beside the onnxruntime.dll its providers were built against.  clpeak does
# not carry either DLL: this script downloads the package (a zip) from
# nuget.org and copies its native DLLs for one architecture, with Microsoft's
# license and third-party notices, into build\winml\<arch>\ (git-ignored).
# DirectML.dll comes along (~19 MB of the ~50 MB package): that runtime
# delay-loads it the moment its DirectML provider is attached, which clpeak's
# viability probe does, and a delay-load with no DLL to find is a crash, not
# a refusal.
#
# Then:
#   build\clpeak.exe --onnx --onnx-winml build\winml\<arch> --list-devices
#
# --onnx-winml with a directory makes the onnxruntime.dll beside the catalog
# the runtime as well (no --onnx-lib needed), and the first enumeration
# installs whatever certified providers fit the machine -- a download that
# can take a few minutes.  The GUI's Settings has the same switch.
param(
    [string]$Version = "2.3.42",
    [ValidateSet("x64", "arm64", "")]
    [string]$Arch = "",
    [string]$Dest = ""
)

$ErrorActionPreference = "Stop"

if ($Arch -eq "") {
    $Arch = if ($env:PROCESSOR_ARCHITECTURE -eq "ARM64") { "arm64" } else { "x64" }
}
$root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
if ($Dest -eq "") {
    $Dest = Join-Path $root "build\winml\$Arch"
}

$package = "Microsoft.Windows.AI.MachineLearning"
$url = "https://www.nuget.org/api/v2/package/$package/$Version"
$staging = Join-Path ([System.IO.Path]::GetTempPath()) "clpeak-winml-$([System.Guid]::NewGuid().ToString('N'))"
New-Item -ItemType Directory -Path $staging | Out-Null
try {
    $zip = Join-Path $staging "$package.$Version.zip"
    Write-Host "Fetching $package $Version from nuget.org..."
    Invoke-WebRequest -Uri $url -OutFile $zip -UseBasicParsing
    Expand-Archive -Path $zip -DestinationPath (Join-Path $staging "pkg")

    $native = Join-Path $staging "pkg\runtimes\win-$Arch\native"
    if (-not (Test-Path $native)) {
        throw "the package carries no runtimes\win-$Arch\native directory"
    }
    New-Item -ItemType Directory -Path $Dest -Force | Out-Null
    foreach ($name in @("Microsoft.Windows.AI.MachineLearning.dll", "onnxruntime.dll", "DirectML.dll")) {
        Copy-Item (Join-Path $native $name) (Join-Path $Dest $name) -Force
        Write-Host "staged $name"
    }
    foreach ($name in @("license.txt", "ThirdPartyNotices.txt")) {
        $src = Join-Path $staging "pkg\$name"
        if (Test-Path $src) { Copy-Item $src (Join-Path $Dest $name) -Force }
    }
    Write-Host ""
    Write-Host "Windows ML $Version ($Arch) is in $Dest. Try:"
    Write-Host "  build\clpeak.exe --onnx --onnx-winml `"$Dest`" --list-devices"
}
finally {
    Remove-Item -Recurse -Force $staging -ErrorAction SilentlyContinue
}
