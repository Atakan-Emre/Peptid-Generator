"""
Markdown to PDF converter with images using fpdf2
"""
import os
import re
from fpdf import FPDF

PROJECT_ROOT = r"d:\GoogleDrive\Projeler\PeptidGenerator"
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results", "peptide_generation_comparison")
REPORTS_DIR = os.path.join(RESULTS_DIR, "reports")
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")

class MarkdownPDF(FPDF):
    def __init__(self):
        super().__init__()
        self.add_font("DejaVu", "", r"C:\Windows\Fonts\arial.ttf", uni=True)
        self.add_font("DejaVu", "B", r"C:\Windows\Fonts\arialbd.ttf", uni=True)
        self.set_auto_page_break(auto=True, margin=15)
        
    def header(self):
        pass
        
    def footer(self):
        self.set_y(-15)
        self.set_font("DejaVu", "", 8)
        self.set_text_color(128)
        self.cell(0, 10, f"Sayfa {self.page_no()}", align="C")

    def chapter_title(self, title, level=1):
        sizes = {1: 18, 2: 14, 3: 12}
        self.set_font("DejaVu", "B", sizes.get(level, 12))
        colors = {1: (26, 95, 122), 2: (45, 139, 186), 3: (68, 68, 68)}
        self.set_text_color(*colors.get(level, (0, 0, 0)))
        self.multi_cell(0, 8, title)
        if level == 1:
            self.set_draw_color(26, 95, 122)
            self.line(10, self.get_y(), 200, self.get_y())
        self.ln(3)
        self.set_text_color(0)

    def body_text(self, text):
        self.set_font("DejaVu", "", 10)
        # Handle inline code
        text = re.sub(r'`([^`]+)`', r'[\1]', text)
        # Handle bold
        parts = re.split(r'\*\*([^*]+)\*\*', text)
        for i, part in enumerate(parts):
            if i % 2 == 1:
                self.set_font("DejaVu", "B", 10)
                self.write(5, part)
                self.set_font("DejaVu", "", 10)
            else:
                self.write(5, part)
        self.ln(6)

    def add_table(self, headers, rows):
        self.set_font("DejaVu", "B", 9)
        col_widths = [190 / len(headers)] * len(headers)
        
        # Header
        self.set_fill_color(26, 95, 122)
        self.set_text_color(255)
        for i, h in enumerate(headers):
            self.cell(col_widths[i], 7, h.strip(), border=1, fill=True, align="C")
        self.ln()
        
        # Rows
        self.set_font("DejaVu", "", 9)
        self.set_text_color(0)
        fill = False
        for row in rows:
            if fill:
                self.set_fill_color(245, 245, 245)
            else:
                self.set_fill_color(255, 255, 255)
            for i, cell in enumerate(row):
                cell_text = cell.strip().replace('`', '')
                self.cell(col_widths[i], 6, cell_text[:30], border=1, fill=True, align="C")
            self.ln()
            fill = not fill
        self.ln(3)

    def add_image(self, img_path, base_path):
        # Resolve relative path
        if img_path.startswith("../"):
            full_path = os.path.normpath(os.path.join(base_path, img_path))
        else:
            full_path = img_path
        
        if os.path.exists(full_path):
            # Check if we need a new page
            if self.get_y() > 200:
                self.add_page()
            self.image(full_path, x=10, w=190)
            self.ln(5)
        else:
            self.set_font("DejaVu", "", 9)
            self.set_text_color(255, 0, 0)
            self.cell(0, 5, f"[Gorsel bulunamadi: {img_path}]")
            self.set_text_color(0)
            self.ln(3)

def md_to_pdf(md_file, pdf_file=None):
    if pdf_file is None:
        pdf_file = md_file.replace(".md", ".pdf")
    
    base_path = os.path.dirname(md_file)
    
    with open(md_file, "r", encoding="utf-8") as f:
        lines = f.readlines()
    
    pdf = MarkdownPDF()
    pdf.add_page()
    
    in_table = False
    table_headers = []
    table_rows = []
    in_code_block = False
    
    for line in lines:
        line = line.rstrip()
        
        # Code block
        if line.startswith("```"):
            in_code_block = not in_code_block
            continue
        
        if in_code_block:
            pdf.set_font("DejaVu", "", 8)
            pdf.set_fill_color(244, 244, 244)
            pdf.cell(0, 5, line, fill=True)
            pdf.ln()
            continue
        
        # Headers
        if line.startswith("# "):
            pdf.chapter_title(line[2:].replace("🧬", "").replace("📋", "").replace("📊", "").replace("🏆", "").replace("📝", "").replace("🔬", "").replace("🔗", "").replace("📁", "").replace("📌", "").strip(), 1)
        elif line.startswith("## "):
            pdf.chapter_title(line[3:].replace("🧬", "").replace("📋", "").replace("📊", "").replace("🏆", "").replace("📝", "").replace("🔬", "").replace("🔗", "").replace("📁", "").replace("📌", "").strip(), 2)
        elif line.startswith("### "):
            pdf.chapter_title(line[4:].strip(), 3)
        
        # Table
        elif line.startswith("|"):
            cells = [c.strip() for c in line.split("|")[1:-1]]
            if all(c.replace("-", "").replace(":", "") == "" for c in cells):
                continue  # Separator line
            elif not in_table:
                in_table = True
                table_headers = cells
            else:
                table_rows.append(cells)
        elif in_table and not line.startswith("|"):
            if table_headers and table_rows:
                pdf.add_table(table_headers, table_rows)
            in_table = False
            table_headers = []
            table_rows = []
        
        # Image
        elif line.startswith("!["):
            match = re.search(r'\!\[.*?\]\((.*?)\)', line)
            if match:
                pdf.add_image(match.group(1), base_path)
        
        # Horizontal rule
        elif line.startswith("---"):
            pdf.ln(3)
            pdf.set_draw_color(200)
            pdf.line(10, pdf.get_y(), 200, pdf.get_y())
            pdf.ln(5)
        
        # List items
        elif line.startswith("- "):
            pdf.set_font("DejaVu", "", 10)
            pdf.cell(5, 5, chr(149))  # bullet
            pdf.body_text(line[2:])
        elif re.match(r'^\d+\.', line):
            pdf.body_text(line)
        
        # Regular text
        elif line.strip() and not line.startswith("*"):
            pdf.body_text(line)
    
    # Flush remaining table
    if in_table and table_headers and table_rows:
        pdf.add_table(table_headers, table_rows)
    
    pdf.output(pdf_file)
    print(f"✓ PDF olusturuldu: {pdf_file}")
    return pdf_file

if __name__ == "__main__":
    import sys
    plastic = sys.argv[1] if len(sys.argv) > 1 else "PET"
    md_file = os.path.join(REPORTS_DIR, f"RAPOR_{plastic}.md")
    pdf_file = os.path.join(REPORTS_DIR, f"RAPOR_{plastic}.pdf")
    
    if os.path.exists(md_file):
        md_to_pdf(md_file, pdf_file)
    else:
        print(f"Dosya bulunamadi: {md_file}")
