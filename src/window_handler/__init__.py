from .window_handler_abstract import WindowHandlerAbstract
import platform
import os


def create_window_handler(win_name: str, x11_name: str) -> WindowHandlerAbstract:
    system_name = platform.system()

    if system_name == 'Windows':
        from .windows_handler import WindowsHandler
        return WindowsHandler(win_name)
    elif system_name == 'Linux':
        display_server = os.environ.get('XDG_SESSION_TYPE', 'unknown')
        if display_server == 'x11':
            from .x11_handler import X11Handler
            return X11Handler(x11_name)
        else:
            raise EnvironmentError(f"Unsupported display server: {display_server}.")
    else:
        raise EnvironmentError(f"Unsupported OS : {system_name}.")
