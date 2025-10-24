from urllib.parse import urlencode
from django.conf import settings
from django.shortcuts import redirect
from django.urls import reverse
from django.utils.deprecation import MiddlewareMixin

SAFE_PATHS_PREFIXES = (
    "/gate",         # allow the password page
    "/admin/login",  # allow admin login page
    "/static/",      # allow static assets
    "/favicon.ico",
)

class PasswordGateMiddleware(MiddlewareMixin):
    def process_request(self, request):
        # Allow safe/bypass paths
        path = request.path
        if path.startswith(SAFE_PATHS_PREFIXES):
            return None

        # Already unlocked?
        if request.session.get("passed_gate", False):
            return None

        # Otherwise, redirect to gate and preserve 'next'
        q = urlencode({"next": request.get_full_path()})
        return redirect(f"{reverse('gate')}?{q}")
