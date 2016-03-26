/*
 * Courtesy: https://gist.github.com/sebclaeys/1227644
 */


#ifndef XML_WRITER_H
# define XML_WRITER_H

# define HEADER "<?xml version=\"1.0\" encoding=\"utf-8\"?>\n"
# define INDENT "  "
# define NEWLINE "\n"

# include <string>
# include <stack>
# include <iostream>

class xmlWriter
{
public:

  xmlWriter(std::ostream& os) : os(os), tag_open(false), new_line(true) {os << HEADER;}
  ~xmlWriter() {}

  xmlWriter& openElt(const char* tag) {
    this->closeTag();
    if (elt_stack.size() > 0)
      os << NEWLINE;
    this->indent();
    this->os << "<" << tag;
    elt_stack.push(tag);
    tag_open = true;
    new_line = false;
    return *this;
  }

  xmlWriter& closeElt() {
    this->closeTag();
    std::string elt = elt_stack.top();
    this->elt_stack.pop();
    if (new_line)
    {
      os << NEWLINE;
      this->indent();
    }
    new_line = true;
    this->os << "</" << elt << ">";
    return *this;
  }

  xmlWriter& closeAll() {
    while (elt_stack.size())
      this->closeElt();
    return *this;
  }

  xmlWriter& attr(const char* key, const char* val) {
    this->os << " " << key << "=\"";
    this->write_escape(val);
    this->os << "\"";
    return *this;
  }

  xmlWriter& attr(const char* key, std::string val) {
    return attr(key, val.c_str());
  }

  xmlWriter& content(const char* val) {
    this->closeTag();
    this->write_escape(val);
    return *this;
  }

private:
  std::ostream& os;
  bool tag_open;
  bool new_line;
  std::stack<std::string> elt_stack;

  inline void closeTag() {
    if (tag_open)
    {
      this->os << ">";
      tag_open = false;
    }
  }

  inline void indent() {
    for (int i = 0; i < (int)elt_stack.size(); i++)
      os << (INDENT);
  }

  inline void write_escape(const char* str) {
    for (; *str; str++)
      switch (*str) {
      case '&': os << "&amp;"; break;
      case '<': os << "&lt;"; break;
      case '>': os << "&gt;"; break;
      case '\'': os << "&apos;"; break;
      case '"': os << "&quot;"; break;
      default: os.put(*str); break;
      }
  }
};

#endif /* !XML_WRITER_H */
