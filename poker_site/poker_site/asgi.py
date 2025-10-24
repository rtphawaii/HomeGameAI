import os
from channels.routing import ProtocolTypeRouter, URLRouter
from django.core.asgi import get_asgi_application
from channels.auth import AuthMiddlewareStack
import poker.routing  # this is your app name

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'poker_site.settings')

application = ProtocolTypeRouter({
    'http': get_asgi_application(),
    'websocket': AuthMiddlewareStack(
        URLRouter(
            poker.routing.websocket_urlpatterns  # must match your routing.py
        )
    ),
})