import os, secrets, uuid
from django.conf import settings
from django.http import JsonResponse, HttpResponseForbidden
from django.shortcuts import render, redirect
from django.urls import reverse

# Gate password comes from settings (or env)
GATE_PASSWORD = getattr(settings, "GATE_PASSWORD", os.environ.get("GATE_PASSWORD", ""))

def gate_view(request):
    if request.session.get("passed_gate"):
        nxt = request.GET.get("next") or reverse("lobby")
        return redirect(nxt)

    error = None
    if request.method == "POST":
        provided = request.POST.get("password", "")
        if GATE_PASSWORD and secrets.compare_digest(provided, GATE_PASSWORD):
            request.session["passed_gate"] = True
            nxt = request.GET.get("next") or reverse("lobby")
            return redirect(nxt)
        error = "Incorrect password."
    return render(request, "poker/gate.html", {"error": error})

def gate_lock(request):
    request.session.pop("passed_gate", None)
    return redirect(reverse("gate"))

def gate_health(request):
    if not GATE_PASSWORD:
        return HttpResponseForbidden("Gate not configured.")
    return redirect(reverse("gate"))

def lobby_view(request):
    # Enforce gate
    if not request.session.get("passed_gate"):
        return redirect(f"{reverse('gate')}?next={request.path}")

    # Give each browser a stable user_id (stored in session)
    if not request.session.get("user_id"):
        request.session["user_id"] = uuid.uuid4().hex
    return render(request, "poker/lobby.html", {"user_id": request.session["user_id"]})

def restart_view(request):
    # Dev-only toy endpoint
    return JsonResponse({"status": "ok"})
