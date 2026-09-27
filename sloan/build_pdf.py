"""Builds sloan/SSAC2027_abstract.pdf from sloan/abstract.md, results/sloan_table1.txt
and figures/sloan_figure1.png. Run from the repository root after script 04."""
import re
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, KeepTogether
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

F = '/usr/share/fonts/truetype/crosextra/'
pdfmetrics.registerFont(TTFont('Carlito', F + 'Carlito-Regular.ttf'))
pdfmetrics.registerFont(TTFont('Carlito-Bold', F + 'Carlito-Bold.ttf'))
pdfmetrics.registerFont(TTFont('Carlito-Italic', F + 'Carlito-Italic.ttf'))
from reportlab.pdfbase.pdfmetrics import registerFontFamily
registerFontFamily('Carlito', normal='Carlito', bold='Carlito-Bold', italic='Carlito-Italic')

NAVY = colors.HexColor('#16325c')
title = ParagraphStyle('t', fontName='Carlito-Bold', fontSize=17, leading=21, textColor=NAVY, spaceAfter=4)
sub = ParagraphStyle('s', fontName='Carlito', fontSize=10, leading=13, textColor=colors.HexColor('#5b6475'), spaceAfter=10)
h = ParagraphStyle('h', fontName='Carlito-Bold', fontSize=11.5, leading=14, textColor=NAVY, spaceBefore=7, spaceAfter=2)
body = ParagraphStyle('b', fontName='Carlito', fontSize=10.5, leading=14, alignment=0, spaceAfter=4)
cap = ParagraphStyle('c', fontName='Carlito', fontSize=9, leading=11.5, textColor=colors.HexColor('#333333'))
cell = ParagraphStyle('cell', fontName='Carlito', fontSize=9.5, leading=12)

md = open('sloan/abstract.md', encoding='utf-8').read()
md = md.replace('github.com/LucaStubel/mwdt-research',
                '<link href="https://github.com/LucaStubel/mwdt-research" color="#16325c"><u>github.com/LucaStubel/mwdt-research</u></link>')
md = re.sub(r'\*\*(.+?)\*\*', r'<b>\1</b>', md)
blocks = [b.strip() for b in md.split('\n\n') if b.strip()]

story, captions = [], {}
for b in blocks:
    if b.startswith('# '):
        story += [Paragraph(b[2:], title), Spacer(1, 4)]
    elif b.startswith('## '):
        story.append(Paragraph(b[3:], h))
    elif b.startswith('<b>Table 1.'):
        captions['table'] = b
    elif b.startswith('<b>Figure 1.'):
        captions['figure'] = b
    else:
        story.append(Paragraph(b, body))

# Table 1 from results file
lines = open('results/sloan_table1.txt', encoding='utf-8').read().splitlines()
def cells(line):
    parts = re.split(r'\s{2,}', line.strip())
    return parts
data = [['', '(1) MWDT', '(2) + rotation size', '(3) Concentration']]
for line in lines[2:6]:
    p = cells(line)
    p = [re.sub(r' \(p = (\d?\.\d+)\)', lambda m: '<br/><font size="8" color="#5b6475">' + ('p &lt; .001' if float(m.group(1)) < .001 else f'p = {float(m.group(1)):.3f}'.replace('0.', '.')) + '</font>', x) for x in p]
    data.append([Paragraph(x, cell) for x in p])
t = Table(data, colWidths=[2.0 * inch, 1.55 * inch, 1.55 * inch, 1.55 * inch])
t.setStyle(TableStyle([
    ('FONTNAME', (0, 0), (-1, 0), 'Carlito-Bold'), ('FONTSIZE', (0, 0), (-1, 0), 9.5),
    ('TEXTCOLOR', (0, 0), (-1, 0), NAVY),
    ('LINEABOVE', (0, 0), (-1, 0), 1, NAVY), ('LINEBELOW', (0, 0), (-1, 0), .6, NAVY),
    ('LINEBELOW', (0, -1), (-1, -1), 1, NAVY), ('LINEABOVE', (0, -1), (-1, -1), .3, colors.grey),
    ('ALIGN', (1, 0), (-1, -1), 'CENTER'), ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
    ('TOPPADDING', (0, 0), (-1, -1), 3), ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
]))
for r in range(1, len(data)):
    for c in range(1, 4):
        data[r][c].style = ParagraphStyle('cc', parent=cell, alignment=1)

story += [Spacer(1, 8), KeepTogether([Paragraph(captions['table'], cap), Spacer(1, 3), t]), Spacer(1, 10)]
img = Image('figures/sloan_figure1.png', width=6.9 * inch, height=6.9 * inch * 3.9 / 10.2)
story += [KeepTogether([img, Spacer(1, 3), Paragraph(captions['figure'], cap)])]

doc = SimpleDocTemplate('sloan/SSAC2027_abstract.pdf', pagesize=letter, leftMargin=.8 * inch, rightMargin=.8 * inch,
                        topMargin=.7 * inch, bottomMargin=.7 * inch,
                        title='Chemistry or Rotation Size? Auditing Pair-Level Familiarity Metrics in the NBA')
doc.build(story)
print('Saved sloan/SSAC2027_abstract.pdf')
