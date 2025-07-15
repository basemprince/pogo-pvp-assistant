"""Utility functions and helpers for the Poké PvP assistant."""

import csv
import io
import json
import os
import pickle
import re
import shutil
import sys
import time
import tkinter as tk
from datetime import datetime
from difflib import get_close_matches
from pathlib import Path
from tkinter import messagebox

import cv2
import numpy as np
import pandas as pd
import requests
import yaml
from PIL import Image
from skimage.color import rgb2lab

# Colors used for Pokemon typings throughout the application
TYPING_HEX_COLORS = {
    "normal": "#a0a29f",
    "fire": "#fba64c",
    "water": "#539ddf",
    "electric": "#f2d94e",
    "grass": "#60bd58",
    "ice": "#76d1c1",
    "fighting": "#d3425f",
    "poison": "#b763cf",
    "ground": "#da7c4d",
    "flying": "#a1bbec",
    "psychic": "#fa8582",
    "bug": "#92bd2d",
    "rock": "#c9bc8a",
    "ghost": "#5f6dbc",
    "dragon": "#0c6ac8",
    "dark": "#595761",
    "steel": "#5795a3",
    "fairy": "#ef90e6",
}

# Paths
BASE_DIR = Path(__file__).resolve().parents[1]
CONFIG_DIR = BASE_DIR / "config"

# isort: off
# pylint: disable=no-name-in-module,wrong-import-order,wrong-import-position,cyclic-import
import scrcpy.scrcpy_python_client as scrcpy  # noqa: E402
from . import roi_ui  # noqa: E402

# isort: on


def load_pokemon_names():
    """Return a dictionary of Pokémon names indexed by ID."""
    with open("json_files/pk.json", "r", encoding="utf-8") as file:
        return json.load(file)


def load_pokemon_details():
    """Return detailed Pokémon information."""
    with open("json_files/pokemon.json", "r", encoding="utf-8") as file:
        return json.load(file)


def load_moves_info():
    """Return move data from the JSON file."""
    with open("json_files/moves.json", "r", encoding="utf-8") as file:
        return json.load(file)


def load_alignment_df(counts=4):
    """Return a dataframe describing move alignment counts."""
    alignment_info = """,1,2,3,4,5
    1,,"1,2","2,3","3,4","4,5"
    2,,,"1,3","1,2","2,5"
    3,,"1,2",,"1,4","3,5"
    4,,,"2,3",,"1,5"
    5,,"1,2","1,4","3,4","""

    df = pd.read_csv(io.StringIO(alignment_info), index_col=0)

    for index, _ in df.iterrows():
        for col in df.columns:
            result = find_correct_alignment(df, index, col, counts)
            df.at[index, col] = result
    return df


def load_phone_data(device_name):
    """Load saved region-of-interest data for ``device_name``."""
    yaml_file = CONFIG_DIR / "phone_roi.yaml"
    with open(yaml_file, "r", encoding="utf-8") as file:
        data = yaml.safe_load(file)
    if data is None:
        return None
    if device_name in data:
        return data[device_name]
    return None


def get_phone_data(client):
    """Ensure phone ROI data exists for the given scrcpy client."""

    timeout = 5
    start = time.time()
    while client.state.resolution is None and time.time() - start < timeout:
        time.sleep(0.05)

    if client.state.resolution is None:
        raise RuntimeError("Timed out waiting for video resolution from scrcpy.")

    phone_data = load_phone_data(client.state.device_name)

    if phone_data is None:

        roi_selector_cls = roi_ui.RoiSelector

        app = roi_selector_cls(client)
        app.update_ui(client)
        app.mainloop()
        phone_data = load_phone_data(client.state.device_name)

    if phone_data:
        roi_dict = {
            roi_key: phone_data.get(roi_key)
            for roi_key in [
                "my_roi",
                "opp_roi",
                "msgs_roi",
                "my_pokeballs_roi",
                "opp_pokeballs_roi",
                "my_typing_roi",
                "opp_typing_roi",
                "first_charge_mv_roi",
                "second_charge_mv_roi",
            ]
        }
    else:
        print("Failed to retrieve phone data")
        return None

    return roi_dict


def find_correct_alignment(df, row, col, counts):
    """Expand alignment ranges found in the CSV snippet."""
    if pd.isna(df.at[row, col]):
        return None
    move_counts = [int(count) for count in df.at[row, col].split(",")]
    if len(move_counts) < 2:
        return None
    first_count, step = move_counts[:2]
    return [first_count + step * i for i in range(counts)]


def update_data(button=True):
    """Update all JSON data files if the user confirms."""
    try:
        with open("json_files/last_update_time.pkl", "rb") as file:
            last_update_time = pickle.load(file)
    except FileNotFoundError:
        last_update_time = None

    if last_update_time and button and (datetime.now() - last_update_time).total_seconds() < 3600:
        print("Cooldown period has not passed. Please try again later.")
        return

    if last_update_time and not button and (datetime.now() - last_update_time).total_seconds() / (3600 * 24) < 7:
        print("json files still relatively new.. skipping update")
        return

    response = (
        messagebox.askyesno(
            "Confirmation",
            "Are you sure you want to update the Pokémon and leagues data from PvPoke?",
        )
        if button
        else True
    )
    if response:
        print("updating json files. please wait ....")
        update_pk_info()
        update_move_info()
        update_leagues_and_cups(True)

        last_update_time = datetime.now()
        with open("json_files/last_update_time.pkl", "wb") as file:
            pickle.dump(last_update_time, file)

        print("Data updated successfully!")
    else:
        print("Update cancelled.")


def update_json_files():  # pylint: disable=too-many-locals
    """Download ranking JSON files from PvPoke and store them locally."""
    try:
        repo_owner = "pvpoke"
        repo_name = "pvpoke"
        folder_path = "src/data/rankings/all/overall/"
        destination_directory = "json_files/rankings"

        # Delete all files in the destination directory
        if os.path.isdir(destination_directory):
            shutil.rmtree(destination_directory)
        os.makedirs(destination_directory)

        headers = {"Accept": "application/vnd.github+json"}
        url = f"https://api.github.com/repos/{repo_owner}/{repo_name}/contents/{folder_path}"
        response = requests.get(url, headers=headers)

        if response.status_code == 200:
            files = response.json()
            for file in files:
                if file["type"] == "file" and file["name"].endswith(".json"):
                    download_url = file["download_url"]
                    file_name = file["name"]

                    if file_name == "rankings-1500.json":
                        new_file_name = "Great League.json"
                    elif file_name == "rankings-2500.json":
                        new_file_name = "Ultra League.json"
                    elif file_name == "rankings-10000.json":
                        new_file_name = "Master League.json"
                    elif file_name == "rankings-500.json":
                        new_file_name = "Little Cup.json"
                    else:
                        new_file_name = file_name

                    local_path = os.path.join(destination_directory, new_file_name)

                    file_content = requests.get(download_url).content
                    with open(local_path, "wb") as f:
                        f.write(file_content)
                        print(f"Downloaded {local_path}")
        else:
            print(f"Failed to get folder content, status code: {response.status_code}")
    except Exception as e:
        print(f"An error occurred: {str(e)}")


def update_leagues_and_cups(update=False):
    """Retrieve available cups and rankings from PvPoke."""
    cup_names_combo_box = ["Great League", "Ultra League", "Master League"]
    save_cup_names = []
    try:
        if update:
            update_json_files()
            avail_cups = download_current_cups()
            for cup in avail_cups:
                cup_names_combo_box.append(cup["title"])
                save_cup_names.append(cup["title"])
            for fmt in avail_cups:
                title = fmt["title"]
                cup = fmt["cup"]
                category = "overall"
                league = fmt["cp"]
                download_ranking_data(cup, category, league, title)
            with open("json_files/saved_cup_names.pkl", "wb") as f:
                pickle.dump(save_cup_names, f)
        else:
            with open("json_files/saved_cup_names.pkl", "rb") as f:
                avail_cups = pickle.load(f)
            cup_names_combo_box.extend(avail_cups)
        return cup_names_combo_box
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        return cup_names_combo_box


def download_ranking_data(cup, category, league, title):  # pylint: disable=too-many-locals
    """Download and save ranking data for the given cup."""
    key = f"{cup}{category}{league}"
    object_rankings = {}
    repo_owner = "pvpoke"
    repo_name = "pvpoke"
    headers = {"Accept": "application/vnd.github+json"}
    if key not in object_rankings:
        file_path = f"src/data/rankings/{cup}/{category}/rankings-{league}.json"
        url = f"https://api.github.com/repos/{repo_owner}/{repo_name}/contents/{file_path}"

        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            file_info = response.json()
            download_url = file_info["download_url"]
            file_content = requests.get(download_url).content

            local_path = os.path.join("json_files/rankings/", f"{title}.json")

            with open(local_path, "wb") as f:
                f.write(file_content)
                print(f"Downloaded {local_path}")
        else:
            print("Failed to get file content")


def update_format_select(formats):
    """Filter and return formats that should be displayed to the user."""
    visible_formats = [
        format
        for format in formats
        if format["showFormat"]
        and not format.get("hideRankings", False)
        and "Silph" not in format["title"]
        and format["title"] != "Custom"
    ]
    return visible_formats


def update_pk_info():
    """Download the latest Pokémon data JSON file."""
    repo_owner = "pvpoke"
    repo_name = "pvpoke"
    file_path = "src/data/gamemaster/pokemon.json"
    destination_directory = "json_files/"

    headers = {"Accept": "application/vnd.github+json"}
    url = f"https://api.github.com/repos/{repo_owner}/{repo_name}/contents/{file_path}"
    response = requests.get(url, headers=headers)

    if response.status_code == 200:
        file_info = response.json()
        download_url = file_info["download_url"]
        file_name = file_info["name"]

        local_path = os.path.join(destination_directory, file_name)

        file_content = requests.get(download_url).content
        with open(local_path, "wb") as f:
            f.write(file_content)
            print(f"Downloaded {local_path}")
    else:
        print("failed")


def update_move_info():
    """Download the latest move data JSON file."""
    repo_owner = "pvpoke"
    repo_name = "pvpoke"
    file_path = "src/data/gamemaster/moves.json"
    destination_directory = "json_files/"

    headers = {"Accept": "application/vnd.github+json"}
    url = f"https://api.github.com/repos/{repo_owner}/{repo_name}/contents/{file_path}"
    response = requests.get(url, headers=headers)

    if response.status_code == 200:
        file_info = response.json()
        download_url = file_info["download_url"]
        file_name = file_info["name"]

        local_path = os.path.join(destination_directory, file_name)

        file_content = requests.get(download_url).content
        with open(local_path, "wb") as f:
            f.write(file_content)
            print(f"Downloaded {local_path}")
    else:
        print("failed")


def download_current_cups():
    """Download cup information from PvPoke and return visible formats."""
    repo_owner = "pvpoke"
    repo_name = "pvpoke"
    file_path = "src/data/gamemaster.json"
    destination_directory = "json_files/rankings"

    headers = {"Accept": "application/vnd.github+json"}
    url = f"https://api.github.com/repos/{repo_owner}/{repo_name}/contents/{file_path}"
    response = requests.get(url, headers=headers)

    if response.status_code == 200:
        file_info = response.json()
        download_url = file_info["download_url"]
        file_name = file_info["name"]

        local_path = os.path.join(destination_directory, file_name)

        file_content = requests.get(download_url).content
        with open(local_path, "wb") as f:
            f.write(file_content)
            print(f"Downloaded {local_path}")
    else:
        return None

    with open(local_path, encoding="utf-8") as f:
        data = json.load(f)

    # Extract formats and call update_format_select
    formats = data["formats"]

    return update_format_select(formats)


def connect_to_device(ip, docker=False):
    """Return a scrcpy client connected to the device."""
    try:
        config_obj = scrcpy.ClientConfig(ip=ip, docker=docker, max_width=3120)
        client = scrcpy.Client(config_obj)
        return client
    except RuntimeError as e:
        if "No adb exe could be found" in str(e):
            print("ADB executable not found. Please install Android platform tools and ensure 'adb' is in your PATH.")
        raise


# pylint: disable=too-many-locals
def get_roi_images(frame, roi_dict):
    """Crop `frame` according to ROI specs (rect or circ)."""
    roi_images = {}

    for roi_name, roi in roi_dict.items():
        roi_type = roi.get("object_type")
        coords = roi.get("coords")

        if roi_type == "rect":
            x, y, w, h = coords
            roi_images[roi_name] = frame[y : y + h, x : x + w]

        elif roi_type == "circ":
            cx, cy, r = coords

            # Bounding box (clip to frame)
            x1 = max(cx - r, 0)
            y1 = max(cy - r, 0)
            x2 = min(cx + r, frame.shape[1])
            y2 = min(cy + r, frame.shape[0])

            # Crop image
            crop = frame[y1:y2, x1:x2]
            h, w = crop.shape[:2]

            # Mask with validation
            if h > 0 and w > 0:
                mask = np.zeros((h, w), dtype=np.uint8)
                center = (w // 2, h // 2)
                radius = min(w, h) // 2
                if radius > 0:
                    cv2.circle(mask, center, radius, 255, -1)
                else:
                    mask.fill(255)  # If radius is 0, fill entire mask
            else:
                # Skip invalid ROI
                continue

            # Apply mask to each channel
            masked = cv2.bitwise_and(crop, crop, mask=mask)

            roi_images[roi_name] = masked

        else:
            raise ValueError(f"Unsupported ROI type: {roi_type}")

    return roi_images


# pylint: disable=too-many-locals
def draw_display_frames(frame, roi_dict, feed_res, roi_color=(0, 0, 0), roi_thick=12):
    """Draw ROIs (rect or circ) on `frame` and return a resized PIL image."""
    frame_with_rois = frame.copy()

    for roi in roi_dict.values():
        roi_type = roi.get("object_type")
        coords = roi.get("coords")

        if roi_type == "rect":
            x, y, w, h = coords
            cv2.rectangle(frame_with_rois, (x, y), (x + w, y + h), roi_color, roi_thick)

        elif roi_type == "circ":
            cx, cy, r = coords
            cv2.circle(frame_with_rois, (cx, cy), r, roi_color, roi_thick)

        else:
            raise ValueError(f"Unsupported ROI type: {roi_type}")

    resized_image = cv2.resize(frame_with_rois, feed_res, interpolation=cv2.INTER_AREA)
    pil_img = Image.fromarray(resized_image)
    return pil_img


# Function to find the closest Pokémon name
def closest_name(name, names_list):
    """Return the closest match to ``name`` from ``names_list``."""
    matches = get_close_matches(name, names_list, n=1, cutoff=0.6)
    if matches:
        return matches[0]
    return None


def process_image(img):
    """Return a thresholded version of ``img`` alongside the original."""
    prev_img = img.copy()
    gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    blur_img = cv2.GaussianBlur(gray_img, (5, 5), 0)
    _, thresh_img = cv2.threshold(blur_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = np.ones((2, 2), np.uint8)
    thresh_img = cv2.morphologyEx(thresh_img, cv2.MORPH_OPEN, kernel)
    thresh_img = Image.fromarray(thresh_img)
    return prev_img, thresh_img


def mse(image1, image2):
    """Compute the mean squared error between two images."""
    if image1.size == 0 or image2.size == 0:
        return float("inf")
    error = np.sum((image1.astype("float") - image2.astype("float")) ** 2)
    error /= float(image1.shape[0] * image1.shape[1])
    return error


def get_next_filename(directory):
    """Return the next available MP4 filename inside ``directory``."""
    if not os.path.exists(directory):
        os.makedirs(directory)

    files = [f for f in os.listdir(directory) if f.endswith(".mp4")]

    if not files:
        return os.path.join(directory, "1.mp4")

    nums = sorted(int(f.split(".")[0]) for f in files)
    next_num = nums[-1] + 1
    return os.path.join(directory, f"{next_num}.mp4")


class LeagueDetector:
    """Determine the battle league based on Pokémon CP values."""

    def __init__(self):
        self.league = None
        self.league_pok = None

    @staticmethod
    def extract_cp(info):
        """Extract the CP value from an info string."""
        cp = re.search(r"\bCP\s+(\d+)\b", info)
        return int(cp.group(1)) if cp else None

    def set_league_based_on_cp(self, cp):
        """Update ``self.league`` based on the highest CP seen."""
        if cp <= 500:
            self.league = "Little Cup"
        elif cp <= 1500:
            self.league = "Great League"
        elif cp <= 2500:
            self.league = "Ultra League"
        else:
            self.league = "Master League"

    def load_league_json(self):
        """Load the ranking JSON for the determined league."""
        if self.league:
            self.league_pok = f"json_files/rankings/{self.league}.json"
            try:
                with open(self.league_pok, "r", encoding="utf-8") as file:
                    self.league_pok = json.load(file)
                    print(f"Loaded {self.league} JSON data")
            except FileNotFoundError:
                print(f"Failed to load {self.league} JSON data")

    def detect_league(self, my_info, opp_info):
        """Infer the league and load its rankings based on CP values."""
        my_cp = self.extract_cp(my_info)
        opp_cp = self.extract_cp(opp_info)

        if my_cp and opp_cp:
            higher_cp = max(my_cp, opp_cp)
            self.set_league_based_on_cp(higher_cp)
            self.load_league_json()

        return self.league, self.league_pok


def count_pokeballs(image):
    """Return the number of Pokéball icons detected in ``image``."""

    image_rgb = image

    lower_red = np.array([100, 0, 0])
    upper_red = np.array([255, 50, 50])

    mask = cv2.inRange(image_rgb, lower_red, upper_red)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    mask = cv2.erode(mask, kernel, iterations=2)
    mask = cv2.dilate(mask, kernel, iterations=2)

    # Find contours in the thresholded image
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return len(contours), mask


def hex_to_rgb(hex_color):
    """Convert a hex color string to a rgb tuple."""
    hex_color = hex_color.lstrip("#")
    rgb = tuple(int(hex_color[i : i + 2], 16) for i in (0, 2, 4))
    return rgb  # [::-1]


def detect_emblems(image, color_range=30, save_images=False):  # pylint: disable=too-many-locals
    """Detect Pokémon type emblems in an image."""
    hex_colors = TYPING_HEX_COLORS

    color_ranges = {
        pokemon_type: (
            list(map(lambda x: max(0, x - color_range), hex_to_rgb(color))),
            list(map(lambda x: min(255, x + color_range), hex_to_rgb(color))),
        )
        for pokemon_type, color in hex_colors.items()
    }

    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # Enhance contrast
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    # Apply a Median blur
    # gray = cv2.medianBlur(gray, 3)

    # Use adaptive thresholding
    # gray = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

    # gray = cv2.dilate(gray, None, iterations=3)
    # gray = cv2.erode(gray, None, iterations=2)

    if save_images:
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S%f")
        filename = f"debug/gray_{timestamp}.png"
        filename1 = f"debug/gray_{timestamp}_1.png"
        try:
            cv2.imwrite(filename, gray)
            cv2.imwrite(filename1, image)
        except Exception:
            pass

    circles = cv2.HoughCircles(
        gray, cv2.HOUGH_GRADIENT, dp=1, minDist=25, param1=30, param2=13, minRadius=29, maxRadius=32
    )
    img_with_circles = cv2.cvtColor(gray.copy(), cv2.COLOR_GRAY2RGB)
    if circles is not None:
        circles = np.uint16(np.around(circles))
        # Sort the circles by their radius, from largest to smallest
        sorted_circles = sorted(circles[0], key=lambda x: -float(x[2]))
        top_circles = sorted_circles[:2]
        number_of_emblems = len(top_circles)

        for i in range(number_of_emblems):
            a, b, r = top_circles[i][0], top_circles[i][1], top_circles[i][2]

            # Draw the circumference of the circle.
            cv2.circle(img_with_circles, (a, b), r, (0, 255, 0), 2)

            # Draw a small circle (of radius 1) to show the center.
            cv2.circle(img_with_circles, (a, b), 1, (0, 0, 255), 3)

    else:
        return [], img_with_circles

    # Detect each type of emblem
    type_counts: dict[str, int] = {}
    if circles is not None:
        for circle in circles[0, :]:
            # Create an empty mask
            mask = np.zeros_like(image)
            mask = cv2.circle(mask, (circle[0], circle[1]), circle[2], (255, 255, 255), -1)
            masked_image = cv2.bitwise_and(image, mask)
            # if save_images:
            #     timestamp = datetime.now().strftime("%Y%m%d%H%M%S%f")
            #     filename = f'debug/detected_circle_{timestamp}.png'
            #     try:
            #         cv2.imwrite(filename, masked_image)
            #     except Exception as e:
            #         pass

            for pokemon_type, (lower, upper) in color_ranges.items():
                temp_mask = cv2.inRange(masked_image, np.array(lower), np.array(upper))
                pixel_count = np.count_nonzero(temp_mask)
                type_counts[pokemon_type] = pixel_count + type_counts.get(pokemon_type, 0)

                # if save_images:
                #     result = cv2.bitwise_and(image, image, mask=temp_mask)
                #     timestamp = datetime.now().strftime("%Y%m%d%H%M%S%f")
                #     cv2.imwrite(f'debug/{pokemon_type}_detection_{timestamp}.png', result)

            sorted_types = [
                pokemon_type
                for pokemon_type, pixel_count in sorted(type_counts.items(), key=lambda x: x[1], reverse=True)[
                    :number_of_emblems
                ]
            ]
    else:
        sorted_types = []

    return sorted(sorted_types), img_with_circles


class ChargeCircleDetector:  # pylint: disable=too-many-instance-attributes
    """Detect and track the charging circle during battles."""

    def __init__(self, roi_size=125, small_ratio=0.6):
        self.center_history = []
        self.stabilized_center = None
        self.charge_moves_stored = 0
        self.roi_size = roi_size
        self.large_radius = roi_size
        self.small_radius = int(roi_size * small_ratio)
        self.small_ratio = small_ratio
        self.boundary_history = []
        self.last_boundary = None

    def mask_white_pixels(self, image, min_white, max_white):
        """Remove white pixels from the image using the given HSV ranges."""
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        lower_white = np.array(min_white, dtype=np.uint8)
        upper_white = np.array(max_white, dtype=np.uint8)
        mask = cv2.inRange(hsv, lower_white, upper_white)
        inverted_mask = cv2.bitwise_not(mask)
        return cv2.bitwise_and(image, image, mask=inverted_mask)

    def calculate_filled_proportion(self, circle, boundary_row):
        """Return how much of the circle is filled based on the boundary row."""
        boundary_from_bottom = (circle[1] + circle[2]) - boundary_row
        return boundary_from_bottom / (2 * circle[2])

    def detect_energy_boundary_in_full_image(self, image, circle):
        """Detect the charge boundary row within the full image."""
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        abs_sobel_y = np.abs(sobel_y)
        mask = self.create_annular_mask(image.shape, circle, inner_radius_proportion=0.6)
        abs_sobel_y_masked = cv2.bitwise_and(abs_sobel_y, abs_sobel_y, mask=mask)
        return int(np.argmax(np.sum(abs_sobel_y_masked, axis=1)))

    def create_annular_mask(self, image_shape, circle, inner_radius_proportion=0.3):
        """Create a ring mask around the circle to focus detection."""
        mask = np.zeros(image_shape[:2], dtype=np.uint8)
        center = (int(circle[0]), int(circle[1]))
        outer_radius = int(circle[2])
        inner_radius = int(inner_radius_proportion * circle[2])
        cv2.circle(mask, center, outer_radius, (255), -1)
        cv2.circle(mask, center, inner_radius, (0), -1)
        return mask

    def isolate_typing_color(self, image, tolerance=25):
        """Return an image with only the dominant typing color visible using LAB color space."""
        # Convert to LAB color space
        lab_image = rgb2lab(image / 255.0)

        # Create annular mask for small circle area
        center = (image.shape[1] // 2, image.shape[0] // 2)
        annular_mask = np.zeros(image.shape[:2], dtype=np.uint8)
        cv2.circle(annular_mask, center, self.small_radius, 255, -1)

        # Create black/white pixel masks in LAB space
        l_channel = lab_image[:, :, 0]
        black_mask = (l_channel > 15).astype(np.uint8) * 255  # Exclude very dark (L < 15)
        white_mask = (l_channel < 85).astype(np.uint8) * 255  # Exclude very bright (L > 85)

        # Combine all masks
        combined_mask = cv2.bitwise_and(cv2.bitwise_and(annular_mask, black_mask), white_mask)

        # Convert typing colors to LAB
        typing_lab_colors = {}
        for type_name, hex_color in TYPING_HEX_COLORS.items():
            rgb = np.array(hex_to_rgb(hex_color)).reshape(1, 1, 3) / 255.0  # pylint: disable=too-many-function-args
            lab = rgb2lab(rgb)[0, 0]
            typing_lab_colors[type_name] = lab

        # Sample pixels within mask
        mask_indices = np.where(combined_mask > 0)
        if len(mask_indices[0]) == 0:
            print("Detected move type: no_pixels")
            return image

        sampled_lab = lab_image[mask_indices]

        # Find closest LAB color
        best_type = "unknown"
        best_count = 0

        for type_name, target_lab in typing_lab_colors.items():
            distances = np.linalg.norm(sampled_lab - target_lab, axis=1)
            close_pixels = np.sum(distances < tolerance)

            if close_pixels > best_count:
                best_count = close_pixels
                best_type = type_name

        print(f"Detected move type: {best_type}")

        # Return original image to keep circle detection working
        return image

    def match_circle_size(self, image, center):
        """Look for small circle first, default to large if not found."""
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        # Test small circle first
        small_circle = (center[0], center[1], self.small_radius)
        small_mask = np.zeros(gray.shape, dtype=np.uint8)
        cv2.circle(small_mask, (int(center[0]), int(center[1])), self.small_radius, 255, 5)

        # Check if small circle has good edge response
        edges = cv2.Canny(gray, 30, 100)
        small_score = np.sum(cv2.bitwise_and(edges, small_mask)) / (2 * np.pi * self.small_radius)

        # If small circle has decent edge response, use it; otherwise use large
        threshold = 2.0  # Minimum score for small circle
        if small_score > threshold:
            print(f"Small circle detected, score: {small_score:.1f}")
            return (small_circle, "small")

        large_circle = (center[0], center[1], self.large_radius)
        print(f"Large circle used, small score too low: {small_score:.1f}")
        return (large_circle, "large")

    def detect_charge_circles(self, image):
        """Detect the charge circle and return the filled energy proportion."""
        filtered = self.isolate_typing_color(image)

        gray = cv2.cvtColor(filtered, cv2.COLOR_RGB2GRAY)
        gray = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)

        roi_x_offset = roi_y_offset = 0

        if self.stabilized_center:
            x, y = map(int, self.stabilized_center)
            roi_x_offset = max(x - self.roi_size, 0)
            roi_y_offset = max(y - self.roi_size, 0)
            roi = gray[
                roi_y_offset : min(y + self.roi_size, gray.shape[0]),
                roi_x_offset : min(x + self.roi_size, gray.shape[1]),
            ]
        else:
            roi = gray

        # Detect any circle first to get center
        circles = cv2.HoughCircles(
            roi,
            cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=200,
            param1=50,
            param2=25,
            minRadius=int(self.small_radius * 0.8),
            maxRadius=int(self.large_radius * 1.2),
        )

        if circles is None:
            return None, None

        detected_circle = circles[0, 0]
        detected_circle[0] += roi_x_offset
        detected_circle[1] += roi_y_offset

        # Smooth center
        self.center_history.append((detected_circle[0], detected_circle[1]))
        if len(self.center_history) > 5:
            self.center_history.pop(0)

        avg_x = sum(pos[0] for pos in self.center_history) / len(self.center_history)
        avg_y = sum(pos[1] for pos in self.center_history) / len(self.center_history)
        smoothed_center = (avg_x, avg_y)

        if len(self.center_history) >= 3:
            recent_positions = self.center_history[-3:]
            max_deviation = max(abs(pos[0] - avg_x) + abs(pos[1] - avg_y) for pos in recent_positions)
            if max_deviation < 8:
                self.stabilized_center = smoothed_center

        # Match which circle size fits better
        matched_circle, circle_type = self.match_circle_size(image, smoothed_center)

        # Detect the energy boundary line within the matched circle
        raw_boundary = self.detect_energy_boundary_in_full_image(filtered.copy(), matched_circle)

        # Stabilize boundary line
        self.boundary_history.append(raw_boundary)
        if len(self.boundary_history) > 5:
            self.boundary_history.pop(0)

        if self.last_boundary is None:
            boundary_row = raw_boundary
        else:
            # Use smoothed boundary
            smoothed_boundary = sum(self.boundary_history) / len(self.boundary_history)
            boundary_row = int(smoothed_boundary)

        self.last_boundary = boundary_row
        filled_proportion = self.calculate_filled_proportion(matched_circle, boundary_row)

        # Draw both circles - use RGB colors since image is in RGB format
        green_rgb = (0, 255, 0)  # Pure green
        yellow_rgb = (255, 255, 0)  # Yellow (red + green)

        small_color = green_rgb if circle_type == "small" else yellow_rgb
        large_color = green_rgb if circle_type == "large" else yellow_rgb

        cv2.circle(filtered, (int(smoothed_center[0]), int(smoothed_center[1])), self.small_radius, small_color, 3)
        cv2.circle(filtered, (int(smoothed_center[0]), int(smoothed_center[1])), self.large_radius, large_color, 3)

        # Draw the boundary line in red (RGB format)
        cv2.line(filtered, (0, boundary_row), (image.shape[1], boundary_row), (255, 0, 0), 4)

        print(f"Circle: {circle_type}, fill: {filled_proportion:.2f}, boundary: {boundary_row}")

        return filled_proportion, filtered


def record_battle(me, opp, league):
    """Store a summary of the battle into ``battle_records.csv``."""
    filename = "battle_records.csv"
    fieldnames = [
        "timestamp",
        "league",
        "my_pokemon1",
        "my_pokemon2",
        "my_pokemon3",
        "opp_pokemon1",
        "opp_pokemon2",
        "opp_pokemon3",
    ]

    new_record = {"timestamp": datetime.now(), "league": league}

    for i in range(3):
        my_pokemon_index = me.ui_chosen_pk_ind[i]
        opp_pokemon_index = opp.ui_chosen_pk_ind[i]

        my_pokemon = (
            "None" if my_pokemon_index is None else getattr(me.pokemons[i][my_pokemon_index], "species_name", "None")
        )
        opp_pokemon = (
            "None" if opp_pokemon_index is None else getattr(opp.pokemons[i][opp_pokemon_index], "species_name", "None")
        )

        new_record[f"my_pokemon{i+1}"] = my_pokemon
        new_record[f"opp_pokemon{i+1}"] = opp_pokemon

    try:
        with open(filename, "x", newline="", encoding="utf-8") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerow(new_record)
    except FileExistsError:
        with open(filename, "a", newline="", encoding="utf-8") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writerow(new_record)


class TextRedirector:
    """Redirect ``stdout`` to a Tkinter text widget."""

    def __init__(self, widget):
        """Initialize with the widget and store the original ``stdout``."""
        self.widget = widget
        self.original_stdout = sys.stdout

    def write(self, string):
        """Write text to the widget and keep showing the end of it."""
        try:
            if self.widget.winfo_exists():
                self.widget.insert(tk.END, string)
                self.widget.see(tk.END)
        except Exception:
            self.original_stdout.write(string)  # Fallback to original stdout

    def flush(self):
        """Required for file-like interface."""
