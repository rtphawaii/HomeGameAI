from django.shortcuts import render
import uuid
from django.http import JsonResponse
from .consumers import ChatConsumer


try:
    import pokerlib
    print("✅ pokerlib imported")
except ModuleNotFoundError:
    print("❌ still broken")

def index(request):
    user_id = str(uuid.uuid4())  # generate random user ID
    return render(request, 'poker/index.html', {'user_id': user_id})

def restart_app(request):
    # Clear all global game states
    ChatConsumer.players.clear()
    ChatConsumer.pending_inputs.clear()
    ChatConsumer.pending_inputs_all.clear()
    ChatConsumer.game_started = False
    ChatConsumer.player_count = None
    return JsonResponse({"status": "ok"})

from django.shortcuts import render, redirect
from django.urls import reverse

def room_view(request, room_name: str, room_type: str = "human"):
    # Enforce gate
    if not request.session.get("passed_gate"):
        return redirect(f"{reverse('gate')}?next={request.path}")

    user_id = request.session.get("user_id") or uuid.uuid4().hex
    request.session["user_id"] = user_id

    # choose template by type
    template = "poker/index_cpu.html" if room_type == "cpu" else "poker/index.html"

    return render(request, template, {
        "user_id": user_id,
        "room_name": room_name,
        "room_type": room_type,
    })

from django.http import JsonResponse
from django.views.decorators.http import require_POST
from django.views.decorators.csrf import csrf_exempt
from .consumers import clear_all_rooms

@csrf_exempt
@require_POST
def clear_rooms_view(request):
    clear_all_rooms()
    return JsonResponse({"ok": True, "message": "All rooms cleared."})