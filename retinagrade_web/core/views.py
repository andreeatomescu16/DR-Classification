import io
import json
from datetime import date

from django.conf import settings
from django.contrib import messages
from django.contrib.auth.decorators import login_required
from django.core.paginator import Paginator
from django.http import HttpResponse, Http404
from django.shortcuts import get_object_or_404, redirect, render
from django.utils import timezone
from django.views.decorators.http import require_POST

from .forms import HistoryFilterForm, NotesForm, UploadForm
from .models import GRADE_DESCRIPTIONS, GRADE_LABELS, AnalysisSession, Prediction, RateLimit
from .services import PredictionService, ServiceUnavailableError
from .pdf import generate_report


# ── Helpers ───────────────────────────────────────────────────────────────────

def _require_own_prediction(request, pk):
    pred = get_object_or_404(Prediction, pk=pk)
    if pred.user != request.user:
        raise Http404
    return pred


def _check_rate_limit(user) -> bool:
    limit_obj, _ = RateLimit.objects.get_or_create(user=user)
    if not limit_obj.is_allowed(settings.ANALYSE_RATE_LIMIT):
        return False
    limit_obj.increment()
    return True


# ── Dashboard ─────────────────────────────────────────────────────────────────

@login_required
def dashboard(request):
    predictions = Prediction.objects.filter(user=request.user)
    recent = predictions[:10]

    grade_counts = [0, 0, 0, 0, 0]
    for p in predictions:
        if 0 <= p.grade <= 4:
            grade_counts[p.grade] += 1

    context = {
        'total': predictions.count(),
        'recent': recent,
        'grade_counts': json.dumps(grade_counts),
        'grade_labels': json.dumps(list(GRADE_LABELS.values())),
    }
    return render(request, 'core/dashboard.html', context)


# ── Upload & Analyse ──────────────────────────────────────────────────────────

@login_required
def analyse(request):
    form = UploadForm()
    if request.method == 'POST':
        form = UploadForm(request.POST, request.FILES)
        if form.is_valid():
            if not _check_rate_limit(request.user):
                messages.error(request, 'Rate limit reached: max 20 analyses per hour.')
                return render(request, 'core/analyse.html', {'form': form})

            image_file = form.cleaned_data['image']
            service = PredictionService()
            try:
                result = service.predict(image_file)
            except ServiceUnavailableError as exc:
                return render(request, 'core/service_unavailable.html', {'error': str(exc)}, status=503)

            image_file.seek(0)
            prediction = Prediction.objects.create(
                user=request.user,
                image=image_file,
                grade=result['grade'],
                label=result['label'],
                confidence=result.get('confidence', max(result['probabilities'])),
                probabilities=result['probabilities'],
                grad_cam_image=result.get('grad_cam_image', ''),
            )
            # Attach to today's default session (create if needed)
            today = timezone.now().date()
            session = AnalysisSession.objects.filter(
                user=request.user,
                title='Default Session',
                created_at__date=today,
            ).first()
            if session is None:
                session = AnalysisSession.objects.create(
                    user=request.user,
                    title='Default Session',
                )
            session.predictions.add(prediction)

            return redirect('core:prediction_detail', pk=prediction.pk)

    return render(request, 'core/analyse.html', {'form': form})


# ── Prediction Detail ─────────────────────────────────────────────────────────

@login_required
def prediction_detail(request, pk):
    prediction = _require_own_prediction(request, pk)
    notes_form = NotesForm(instance=prediction)

    if request.method == 'POST' and request.user.is_clinician():
        notes_form = NotesForm(request.POST, instance=prediction)
        if notes_form.is_valid():
            notes_form.save()
            messages.success(request, 'Notes saved.')
            return redirect('core:prediction_detail', pk=pk)

    probs_with_labels = list(zip(GRADE_LABELS.values(), prediction.probabilities))
    context = {
        'prediction': prediction,
        'notes_form': notes_form,
        'probs_with_labels': probs_with_labels,
        'description': prediction.grade_description(),
        'probs_json': json.dumps(prediction.probabilities),
        'grade_labels_json': json.dumps(list(GRADE_LABELS.values())),
    }
    return render(request, 'core/prediction_detail.html', context)


# ── Save Notes ────────────────────────────────────────────────────────────────

@login_required
@require_POST
def save_notes(request, pk):
    prediction = _require_own_prediction(request, pk)
    if not request.user.is_clinician():
        raise Http404
    form = NotesForm(request.POST, instance=prediction)
    if form.is_valid():
        form.save()
        messages.success(request, 'Notes saved.')
    return redirect('core:prediction_detail', pk=pk)


# ── History ───────────────────────────────────────────────────────────────────

@login_required
def history(request):
    qs = Prediction.objects.filter(user=request.user)
    filter_form = HistoryFilterForm(request.GET or None)

    if filter_form.is_valid():
        grade = filter_form.cleaned_data.get('grade')
        date_from = filter_form.cleaned_data.get('date_from')
        date_to = filter_form.cleaned_data.get('date_to')
        if grade != '' and grade is not None:
            qs = qs.filter(grade=int(grade))
        if date_from:
            qs = qs.filter(created_at__date__gte=date_from)
        if date_to:
            qs = qs.filter(created_at__date__lte=date_to)

    paginator = Paginator(qs, 10)
    page_obj = paginator.get_page(request.GET.get('page'))

    return render(request, 'core/history.html', {
        'page_obj': page_obj,
        'filter_form': filter_form,
    })


# ── Delete Prediction ─────────────────────────────────────────────────────────

@login_required
@require_POST
def delete_prediction(request, pk):
    prediction = _require_own_prediction(request, pk)
    prediction.image.delete(save=False)
    prediction.delete()
    messages.success(request, 'Prediction deleted.')
    return redirect('core:history')


@login_required
@require_POST
def bulk_delete(request):
    ids = request.POST.getlist('selected')
    if ids:
        qs = Prediction.objects.filter(pk__in=ids, user=request.user)
        for p in qs:
            p.image.delete(save=False)
        qs.delete()
        messages.success(request, f'Deleted {len(ids)} prediction(s).')
    return redirect('core:history')


# ── PDF Report ────────────────────────────────────────────────────────────────

@login_required
def download_report(request, pk):
    prediction = _require_own_prediction(request, pk)
    pdf_bytes = generate_report(prediction)
    response = HttpResponse(pdf_bytes, content_type='application/pdf')
    response['Content-Disposition'] = f'attachment; filename="retinagrade_report_{pk}.pdf"'
    return response


# ── Admin Dashboard ───────────────────────────────────────────────────────────

@login_required
def admin_dashboard(request):
    if not request.user.is_admin_role():
        raise Http404

    from django.contrib.auth import get_user_model
    User = get_user_model()

    service = PredictionService()
    backend_up = service.health_check()

    grade_counts = [0, 0, 0, 0, 0]
    for p in Prediction.objects.all():
        if 0 <= p.grade <= 4:
            grade_counts[p.grade] += 1

    users = User.objects.order_by('-date_joined')[:20]

    context = {
        'total_users': User.objects.count(),
        'total_predictions': Prediction.objects.count(),
        'grade_counts': json.dumps(grade_counts),
        'grade_labels': json.dumps(list(GRADE_LABELS.values())),
        'users': users,
        'backend_up': backend_up,
    }
    return render(request, 'core/admin_dashboard.html', context)
