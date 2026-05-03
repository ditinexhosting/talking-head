from flask import Flask

from .routes import health_bp, animate_bp, keyframes_bp


def create_app() -> Flask:
    app = Flask(__name__)

    app.register_blueprint(health_bp)
    app.register_blueprint(animate_bp)
    app.register_blueprint(keyframes_bp)

    return app
