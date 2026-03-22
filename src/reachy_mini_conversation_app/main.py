"""Entrypoint for the Reachy Mini conversation app."""

import os
import sys
import time
import asyncio
import argparse
import threading
from typing import Any, Dict, List, Optional

import gradio as gr
import uvicorn
from fastapi import FastAPI
from fastrtc import Stream
from gradio.utils import get_space
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from reachy_mini import ReachyMini, ReachyMiniApp
from reachy_mini_conversation_app.config import config
from reachy_mini_conversation_app.utils import (
    parse_args,
    setup_logger,
    handle_vision_stuff,
    log_connection_troubleshooting,
)


def update_chatbot(chatbot: List[Dict[str, Any]], response: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Update the chatbot with AdditionalOutputs."""
    chatbot.append(response)
    return chatbot


def main() -> None:
    """Entrypoint for the Reachy Mini conversation app."""
    args, _ = parse_args()
    run(args)


def run(
    args: argparse.Namespace,
    robot: ReachyMini = None,
    app_stop_event: Optional[threading.Event] = None,
    settings_app: Optional[FastAPI] = None,
    instance_path: Optional[str] = None,
) -> None:
    """Run the Reachy Mini conversation app."""
    # Putting these dependencies here makes the dashboard faster to load when the conversation app is installed
    from reachy_mini_conversation_app.moves import MovementManager
    from reachy_mini_conversation_app.console import LocalStream
    from reachy_mini_conversation_app.openai_realtime import OpenaiRealtimeHandler
    from reachy_mini_conversation_app.tools.core_tools import ToolDependencies
    from reachy_mini_conversation_app.audio.head_wobbler import HeadWobbler
    from reachy_mini_conversation_app.tools.grocery_tool import GroceryAssistant
    from reachy_mini_conversation_app.tools.reachy_mini_signs import ReachyChef


    grocery_logic = GroceryAssistant(laptop_ip="192.168.31.5")
    chef_behavior_system = ReachyChef() 
    logger = setup_logger(args.debug)
    logger.info("Starting Reachy Mini Conversation App")

    if args.no_camera and args.head_tracker is not None:
        logger.warning(
            "Head tracking disabled: --no-camera flag is set. "
            "Remove --no-camera to enable head tracking."
        )

    if robot is None:
        try:
            robot_kwargs = {}
            if args.robot_name is not None:
                robot_kwargs["robot_name"] = args.robot_name

            logger.info("Initializing ReachyMini (SDK will auto-detect appropriate backend)")
            robot = ReachyMini(**robot_kwargs)

        except TimeoutError as e:
            logger.error(
                "Connection timeout: Failed to connect to Reachy Mini daemon. "
                f"Details: {e}"
            )
            log_connection_troubleshooting(logger, args.robot_name)
            sys.exit(1)

        except ConnectionError as e:
            logger.error(
                "Connection failed: Unable to establish connection to Reachy Mini. "
                f"Details: {e}"
            )
            log_connection_troubleshooting(logger, args.robot_name)
            sys.exit(1)

        except Exception as e:
            logger.error(
                f"Unexpected error during robot initialization: {type(e).__name__}: {e}"
            )
            logger.error("Please check your configuration and try again.")
            sys.exit(1)

    # Auto-enable Gradio in simulation mode (both MuJoCo for daemon and mockup-sim for desktop app)
    status = robot.client.get_status()
    if isinstance(status, dict):
        simulation_enabled = status.get("simulation_enabled", False)
        mockup_sim_enabled = status.get("mockup_sim_enabled", False)
    else:
        simulation_enabled = getattr(status, "simulation_enabled", False)
        mockup_sim_enabled = getattr(status, "mockup_sim_enabled", False)

    is_simulation = simulation_enabled or mockup_sim_enabled

    if is_simulation and not args.gradio:
        logger.info("Simulation mode detected. Automatically enabling gradio flag.")
        args.gradio = True

    camera_worker, _, vision_manager = handle_vision_stuff(args, robot)

    # start_cooking_vision tool: needs camera frames + either local vision or HTTP VLM
    if camera_worker is not None:
        if vision_manager is not None:
            logger.info(
                "Vision tool start_cooking_vision: local model backend (--local-vision).",
            )
        elif config.VLM_SERVER_URL:
            logger.info(
                "Vision tool start_cooking_vision: HTTP VLM at %s (format=%s)",
                config.VLM_SERVER_URL,
                getattr(config, "VLM_REQUEST_FORMAT", "multipart"),
            )
        else:
            logger.warning(
                "Vision tool start_cooking_vision: no backend yet. Set env VLM_SERVER_URL "
                "or run with --local-vision (install extras: pip install '.[local_vision]').",
            )

    # MJPEG Video stream endpoint
    async def video_feed() -> StreamingResponse:
        """Video streaming generator for the camera worker."""

        async def generate() -> Any:
            while True:
                if camera_worker:
                    jpeg = camera_worker.get_latest_jpeg()
                    if jpeg:
                        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n")
                await asyncio.sleep(0.05)

        return StreamingResponse(generate(), media_type="multipart/x-mixed-replace; boundary=frame")

    if settings_app and camera_worker:
        settings_app.add_api_route("/video_feed", video_feed)

    movement_manager = MovementManager(
        current_robot=robot,
        camera_worker=camera_worker,
    )

    head_wobbler = HeadWobbler(set_speech_offsets=movement_manager.set_speech_offsets)

    deps = ToolDependencies(
        reachy_mini=robot,
        movement_manager=None,
        camera_worker=camera_worker,
        vision_manager=vision_manager,
        head_wobbler=None,
        grocery_assistant=grocery_logic,
        reachy_chef=chef_behavior_system,
    )
    current_file_path = os.path.dirname(os.path.abspath(__file__))
    logger.debug(f"Current file absolute path: {current_file_path}")
    chatbot = gr.Chatbot(
        type="messages",
        resizable=True,
        avatar_images=(
            os.path.join(current_file_path, "images", "user_avatar.png"),
            os.path.join(current_file_path, "images", "reachymini_avatar.png"),
        ),
    )
    logger.debug(f"Chatbot avatar images: {chatbot.avatar_images}")

    handler = OpenaiRealtimeHandler(deps, gradio_mode=args.gradio, instance_path=instance_path)

    # Always initialize LocalStream for robot hardware audio
    stream_manager = LocalStream(
        handler,
        robot,
        settings_app=settings_app,
        instance_path=instance_path,
    )

    if args.gradio:
        api_key_textbox = gr.Textbox(
            label="OPENAI API Key",
            type="password",
            value=os.getenv("OPENAI_API_KEY") if not get_space() else "",
        )

        from reachy_mini_conversation_app.gradio_personality import PersonalityUI

        personality_ui = PersonalityUI()
        personality_ui.create_components()

        stream = Stream(
            handler=handler,
            mode="send-receive",
            modality="audio",
            additional_inputs=[
                chatbot,
                api_key_textbox,
                *personality_ui.additional_inputs_ordered(),
            ],
            additional_outputs=[chatbot],
            additional_outputs_handler=update_chatbot,
            ui_args={"title": "Talk with Reachy Mini"},
        )
        # In Gradio mode, we still use stream_manager for hardware loops but mount Gradio UI
        if not settings_app:
            app = FastAPI()
            app.mount("/static", StaticFiles(directory=os.path.join(current_file_path, "static")), name="static")
        else:
            app = settings_app
            try:
                app.mount("/static", StaticFiles(directory=os.path.join(current_file_path, "static")), name="static")
            except Exception:
                pass

        if camera_worker:
            app.add_api_route("/video_feed", video_feed)

        personality_ui.wire_events(handler, stream.ui)

        app = gr.mount_gradio_app(app, stream.ui, path="/chat")
        
        @app.get("/", include_in_schema=False)
        def _root() -> FileResponse:
            index_file = os.path.join(current_file_path, "static", "index.html")
            return FileResponse(index_file)

    # Each async service → its own thread/loop
    if camera_worker:
        camera_worker.start()
    if vision_manager:
        vision_manager.start()

    def poll_stop_event() -> None:
        """Poll the stop event to allow graceful shutdown."""
        if app_stop_event is not None:
            app_stop_event.wait()

        logger.info("App stop event detected, shutting down...")
        try:
            stream_manager.close()
        except Exception as e:
            logger.error(f"Error while closing stream manager: {e}")

    if app_stop_event:
        threading.Thread(target=poll_stop_event, daemon=True).start()

    try:
        if args.gradio:
            # When running uvicorn, we need to start the hardware loops in the background
            # using a lifespan or a manual task. Since uvicorn.run is blocking, we use 
            # a startup event.
            @app.on_event("startup")
            async def startup_event() -> None:
                stream_manager.start()
            
            uvicorn.run(app, host="0.0.0.0", port=7860)
        else:
            stream_manager.launch()
    except KeyboardInterrupt:
        logger.info("Keyboard interruption in main thread... closing server.")
    finally:
        movement_manager.stop()
        head_wobbler.stop()
        if camera_worker:
            camera_worker.stop()
        if vision_manager:
            vision_manager.stop()

        # Ensure media is explicitly closed before disconnecting
        try:
            robot.media.close()
        except Exception as e:
            logger.debug(f"Error closing media during shutdown: {e}")

        # prevent connection to keep alive some threads
        robot.client.disconnect()
        time.sleep(1)
        logger.info("Shutdown complete.")


class ReachyMiniConversationApp(ReachyMiniApp):  # type: ignore[misc]
    """Reachy Mini Apps entry point for the conversation app."""

    custom_app_url = "http://0.0.0.0:7860/"
    dont_start_webserver = False

    def run(self, reachy_mini: ReachyMini, stop_event: threading.Event) -> None:
        """Run the Reachy Mini conversation app."""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        args, _ = parse_args()

        instance_path = self._get_instance_path().parent
        run(
            args,
            robot=reachy_mini,
            app_stop_event=stop_event,
            settings_app=self.settings_app,
            instance_path=instance_path,
        )


if __name__ == "__main__":
    app = ReachyMiniConversationApp()
    try:
        app.wrapped_run()
    except KeyboardInterrupt:
        app.stop()
