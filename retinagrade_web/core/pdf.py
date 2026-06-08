import base64
import io

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (
    HRFlowable, Image, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle,
)

GRADE_LABELS = ['No DR', 'Mild NPDR', 'Moderate NPDR', 'Severe NPDR', 'Proliferative DR']
ACCENT = colors.HexColor('#6c63ff')
DARK = colors.HexColor('#111118')
LIGHT = colors.HexColor('#e8e8f0')
MUTED = colors.HexColor('#8888aa')


def generate_report(prediction) -> bytes:
    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf,
        pagesize=A4,
        leftMargin=2 * cm,
        rightMargin=2 * cm,
        topMargin=2 * cm,
        bottomMargin=2 * cm,
    )

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        'Title', parent=styles['Title'],
        fontSize=18, textColor=ACCENT, spaceAfter=4,
    )
    heading_style = ParagraphStyle(
        'Heading', parent=styles['Heading2'],
        fontSize=12, textColor=DARK, spaceAfter=4,
    )
    body_style = ParagraphStyle(
        'Body', parent=styles['Normal'],
        fontSize=10, textColor=DARK, spaceAfter=6, leading=14,
    )
    muted_style = ParagraphStyle(
        'Muted', parent=styles['Normal'],
        fontSize=8, textColor=MUTED, spaceAfter=4,
    )
    footer_style = ParagraphStyle(
        'Footer', parent=styles['Normal'],
        fontSize=8, textColor=MUTED, alignment=1,
    )

    story = []

    # Header
    story.append(Paragraph('RetinaGrade', title_style))
    story.append(Paragraph('Diabetic Retinopathy Analysis Report', heading_style))
    story.append(HRFlowable(width='100%', thickness=1, color=ACCENT))
    story.append(Spacer(1, 0.4 * cm))

    # Patient info
    info_data = [
        ['Patient', prediction.user.get_full_name() or prediction.user.username],
        ['Username', prediction.user.username],
        ['Date & Time', prediction.created_at.strftime('%Y-%m-%d %H:%M UTC')],
        ['Report ID', f'RG-{prediction.pk:06d}'],
    ]
    info_table = Table(info_data, colWidths=[4 * cm, 13 * cm])
    info_table.setStyle(TableStyle([
        ('FONTSIZE', (0, 0), (-1, -1), 9),
        ('TEXTCOLOR', (0, 0), (0, -1), MUTED),
        ('TEXTCOLOR', (1, 0), (1, -1), DARK),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
    ]))
    story.append(info_table)
    story.append(Spacer(1, 0.4 * cm))
    story.append(HRFlowable(width='100%', thickness=0.5, color=colors.HexColor('#ddddee')))
    story.append(Spacer(1, 0.4 * cm))

    # Images side by side
    images_row = []
    fundus_img = _load_image(prediction.image.path, 7 * cm, 7 * cm)
    if fundus_img:
        images_row.append(fundus_img)
    if prediction.grad_cam_image:
        gcam_img = _load_b64_image(prediction.grad_cam_image, 7 * cm, 7 * cm)
        if gcam_img:
            images_row.append(gcam_img)

    if images_row:
        captions = [Paragraph('Fundus Image', muted_style)]
        if len(images_row) > 1:
            captions.append(Paragraph('Grad-CAM Overlay', muted_style))
        img_table = Table([images_row, captions], colWidths=[8.5 * cm] * len(images_row))
        img_table.setStyle(TableStyle([
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ]))
        story.append(img_table)
        story.append(Spacer(1, 0.4 * cm))

    # Grade result
    story.append(Paragraph('Classification Result', heading_style))
    grade_data = [
        ['Grade', f'{prediction.grade} — {prediction.label}'],
        ['Confidence', f'{prediction.confidence_pct()}%'],
    ]
    grade_table = Table(grade_data, colWidths=[4 * cm, 13 * cm])
    grade_table.setStyle(TableStyle([
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('TEXTCOLOR', (0, 0), (0, -1), MUTED),
        ('TEXTCOLOR', (1, 0), (1, -1), DARK),
        ('FONTNAME', (1, 0), (1, 0), 'Helvetica-Bold'),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
    ]))
    story.append(grade_table)
    story.append(Spacer(1, 0.3 * cm))

    # Probability table
    story.append(Paragraph('Class Probabilities', heading_style))
    prob_header = [['Grade', 'Probability', 'Bar']]
    prob_rows = []
    for i, (lbl, p) in enumerate(zip(GRADE_LABELS, prediction.probabilities)):
        pct = round(p * 100, 1)
        bar = '█' * int(pct / 5) + '░' * (20 - int(pct / 5))
        row = [lbl, f'{pct}%', bar]
        prob_rows.append(row)
    prob_table = Table(prob_header + prob_rows, colWidths=[4 * cm, 2.5 * cm, 10.5 * cm])
    prob_style = [
        ('BACKGROUND', (0, 0), (-1, 0), ACCENT),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f5f5fb')]),
        ('GRID', (0, 0), (-1, -1), 0.25, colors.HexColor('#ddddee')),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
    ]
    # Highlight predicted grade row
    highlight_row = prediction.grade + 1
    prob_style.append(('TEXTCOLOR', (0, highlight_row), (1, highlight_row), ACCENT))
    prob_style.append(('FONTNAME', (0, highlight_row), (1, highlight_row), 'Helvetica-Bold'))
    prob_table.setStyle(TableStyle(prob_style))
    story.append(prob_table)
    story.append(Spacer(1, 0.4 * cm))

    # Clinical description
    story.append(Paragraph('Clinical Interpretation', heading_style))
    story.append(Paragraph(prediction.grade_description(), body_style))

    # Clinical notes (if any)
    if prediction.notes:
        story.append(Spacer(1, 0.3 * cm))
        story.append(Paragraph('Clinician Notes', heading_style))
        story.append(Paragraph(prediction.notes, body_style))

    story.append(Spacer(1, 1 * cm))
    story.append(HRFlowable(width='100%', thickness=0.5, color=colors.HexColor('#ddddee')))
    story.append(Spacer(1, 0.3 * cm))
    story.append(Paragraph(
        'This report is for research purposes only. Not a clinical diagnosis. '
        'Always consult a qualified ophthalmologist for medical advice.',
        footer_style,
    ))

    doc.build(story)
    return buf.getvalue()


def _load_image(path, max_w, max_h):
    try:
        img = Image(path, width=max_w, height=max_h, kind='proportional')
        return img
    except Exception:
        return None


def _load_b64_image(b64_str, max_w, max_h):
    try:
        raw = base64.b64decode(b64_str)
        buf = io.BytesIO(raw)
        img = Image(buf, width=max_w, height=max_h, kind='proportional')
        return img
    except Exception:
        return None
