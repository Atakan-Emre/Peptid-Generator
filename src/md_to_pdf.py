import os
import markdown
import base64
from fpdf import FPDF
from PIL import Image

PROJECT_ROOT = r"d:\GoogleDrive\Projeler\PeptidGenerator"
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results", "peptide_generation_comparison")
REPORTS_DIR = os.path.join(RESULTS_DIR, "reports")
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")

def embed_images(html_content, base_path):
    """Replace image paths with base64 encoded images"""
    import re
    
    def replace_img(match):
        img_path = match.group(1)
        # Handle relative paths
        if img_path.startswith("../"):
            full_path = os.path.normpath(os.path.join(base_path, img_path))
        else:
            full_path = img_path
        
        if os.path.exists(full_path):
            with open(full_path, "rb") as f:
                img_data = base64.b64encode(f.read()).decode()
            ext = os.path.splitext(full_path)[1].lower()
            mime = {"png": "image/png", "jpg": "image/jpeg", "jpeg": "image/jpeg", "gif": "image/gif"}.get(ext[1:], "image/png")
            return f'<img src="data:{mime};base64,{img_data}" style="max-width:100%; height:auto;">'
        return match.group(0)
    
    html_content = re.sub(r'<img[^>]*src="([^"]+)"[^>]*>', replace_img, html_content)
    return html_content

def md_to_pdf(md_file, pdf_file=None):
    """Convert markdown file to PDF with embedded images"""
    
    if pdf_file is None:
        pdf_file = md_file.replace(".md", ".pdf")
    
    # Read markdown
    with open(md_file, "r", encoding="utf-8") as f:
        md_content = f.read()
    
    # Convert to HTML
    html_content = markdown.markdown(
        md_content, 
        extensions=["tables", "fenced_code", "codehilite"]
    )
    
    # CSS styling
    css = """
    @page {
        size: A4;
        margin: 2cm;
    }
    body {
        font-family: 'Segoe UI', Arial, sans-serif;
        font-size: 11pt;
        line-height: 1.5;
        color: #333;
    }
    h1 {
        color: #1a5f7a;
        border-bottom: 3px solid #1a5f7a;
        padding-bottom: 10px;
        font-size: 24pt;
    }
    h2 {
        color: #2d8bba;
        border-bottom: 1px solid #ddd;
        padding-bottom: 5px;
        margin-top: 25px;
        font-size: 16pt;
    }
    h3 {
        color: #444;
        margin-top: 20px;
        font-size: 13pt;
    }
    table {
        border-collapse: collapse;
        width: 100%;
        margin: 15px 0;
        font-size: 10pt;
    }
    th, td {
        border: 1px solid #ddd;
        padding: 8px;
        text-align: left;
    }
    th {
        background-color: #1a5f7a;
        color: white;
        font-weight: bold;
    }
    tr:nth-child(even) {
        background-color: #f9f9f9;
    }
    tr:hover {
        background-color: #f1f1f1;
    }
    code {
        background-color: #f4f4f4;
        padding: 2px 6px;
        border-radius: 3px;
        font-family: 'Consolas', monospace;
        font-size: 10pt;
    }
    pre {
        background-color: #f4f4f4;
        padding: 15px;
        border-radius: 5px;
        overflow-x: auto;
    }
    img {
        max-width: 100%;
        height: auto;
        margin: 15px 0;
        border: 1px solid #ddd;
        border-radius: 5px;
    }
    hr {
        border: none;
        border-top: 2px solid #eee;
        margin: 25px 0;
    }
    strong {
        color: #1a5f7a;
    }
    ul, ol {
        margin: 10px 0;
        padding-left: 25px;
    }
    li {
        margin: 5px 0;
    }
    """
    
    # Full HTML
    full_html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <style>{css}</style>
    </head>
    <body>
        {html_content}
    </body>
    </html>
    """
    
    # Embed images
    base_path = os.path.dirname(md_file)
    full_html = embed_images(full_html, base_path)
    
    # Generate PDF
    HTML(string=full_html).write_pdf(pdf_file)
    
    print(f"✓ PDF oluşturuldu: {pdf_file}")
    return pdf_file

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        plastic = sys.argv[1]
    else:
        plastic = "PET"
    
    md_file = os.path.join(REPORTS_DIR, f"RAPOR_{plastic}.md")
    pdf_file = os.path.join(REPORTS_DIR, f"RAPOR_{plastic}.pdf")
    
    if os.path.exists(md_file):
        md_to_pdf(md_file, pdf_file)
    else:
        print(f"❌ Dosya bulunamadı: {md_file}")
