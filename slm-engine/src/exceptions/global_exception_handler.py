from flask import jsonify
from exceptions.app_exception import AppException

def register_error_handlers(app):

    @app.errorhandler(AppException)
    def handle_custom_exception(error):
        response = jsonify({"code": error.code, "message": error.message})
        response.status_code = 200
        return response

    @app.errorhandler(404)
    def handle_not_found(error):
        return jsonify({"code": 404, "message": "Resource not found"}), 404

    @app.errorhandler(500)
    def handle_internal_error(error):
        return jsonify({"code": 500, "message": error.message}), 500