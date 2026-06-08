from django import template

register = template.Library()

GRADE_COLORS = {0: 'grade-0', 1: 'grade-1', 2: 'grade-2', 3: 'grade-3', 4: 'grade-4'}


@register.filter
def grade_class(grade):
    return GRADE_COLORS.get(int(grade), 'grade-0')


@register.filter
def pct(value):
    try:
        return f'{float(value) * 100:.1f}%'
    except (TypeError, ValueError):
        return '—'


@register.filter
def mul100(value):
    try:
        return round(float(value) * 100, 1)
    except (TypeError, ValueError):
        return 0
