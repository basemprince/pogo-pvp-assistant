#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import cv2
import numpy as np
import time
from Quartz import CGWindowListCopyWindowInfo, kCGWindowListOptionOnScreenOnly, kCGWindowListExcludeDesktopElements, kCGNullWindowID
from mss import mss
# import matplotlib.pyplot as plt
# %matplotlib inline
from AppKit import NSWindow, NSTitledWindowMask
import json
from difflib import get_close_matches
import re
import math
from IPython.display import clear_output
from adbutils import adb
import scrcpy.core as scrcpy
import sys
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QLabel, QPushButton
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import QTimer, Qt
import pandas as pd
import io
from tesserocr import PyTessBaseAPI, PSM
from PIL import Image
import os


# In[ ]:


phones = ['Pixel 3 XL', 'Pixel 7 Pro']
# Load the JSON files
with open('json_files/pk.json', 'r') as file:
    pokemon_names = json.load(file)

with open('json_files/moves.json', 'r') as file:
    moves = json.load(file)

alingment_info = """,1,2,3,4,5
1,,"1,2","2,3","3,4","4,5"
2,,,"1,3","1,2","2,5"
3,,"1,2",,"1,4","3,5"
4,,,"2,3",,"1,5"
5,,"1,2","1,4","3,4","""

df = pd.read_csv(io.StringIO(alingment_info), index_col=0)
def find_correct_first_three_counts(row, col):
    if pd.isna(df.at[row, col]):
        return None
    move_counts = [int(count) for count in df.at[row, col].split(',')]
    first_count, step = move_counts
    return [first_count + step * i for i in range(3)]

for index, row in df.iterrows():
    for col in df.columns:
        result = find_correct_first_three_counts(index, col)
        df.at[index, col] = result

df


# In[ ]:


adb.connect("127.0.0.1:5037")
client = scrcpy.Client(device=adb.device_list()[0])
client.start(threaded=True)
print(client.device_name)
phone_t = phones.index(client.device_name)


# In[ ]:


# Find the scrcpy window
def find_scrcpy_window(owner_name):
    window_info_list = CGWindowListCopyWindowInfo(kCGWindowListOptionOnScreenOnly | kCGWindowListExcludeDesktopElements, kCGNullWindowID)
    for window_info in window_info_list:
        window_owner_name = window_info.get('kCGWindowName', '')
        for i,owner in enumerate(owner_name):
            if owner in window_owner_name:
                print(f'Using phone:{owner}')
                return window_info,i
    return None,0

# Capture the scrcpy window
def capture_scrcpy_window(window, title_bar_height):
    x = window['kCGWindowBounds']['X']
    y = window['kCGWindowBounds']['Y']
    width = window['kCGWindowBounds']['Width']
    height = window['kCGWindowBounds']['Height']
    header_height = title_bar_height  # Height of the window header on macOS

    with mss() as sct:
        monitor = {'left': int(x), 'top': int(y) + header_height, 'width': int(width), 'height': int(height) - header_height}
        screenshot = sct.grab(monitor)
        screenshot_np = np.array(screenshot)

    return screenshot_np

def get_title_bar_height():
    window = NSWindow.alloc().initWithContentRect_styleMask_backing_defer_(
        ((0, 0), (100, 100)),
        NSTitledWindowMask,
        2,
        False
    )
    content_height = window.contentView().frame().size.height
    window_height = window.frame().size.height
    title_bar_height = window_height - content_height
    return title_bar_height

def crop_top(image, ratio):
    height, _, _ = image.shape
    cropped_height = int(height * ratio)
    return image[:cropped_height, :, :]

# Function to find the closest Pokémon name
def closest_pokemon_name(name, names_list):
    closest_name = get_close_matches(name, names_list, n=1, cutoff=0.6)
    if closest_name:
        return closest_name[0]
    return None

def calculate_move_counts(fast_move, charged_move):
    counts = []
    counts.append(math.ceil((charged_move['energy'] * 1) / fast_move['energyGain']))
    counts.append(math.ceil((charged_move['energy'] * 2) / fast_move['energyGain']) - counts[0])
    counts.append(math.ceil((charged_move['energy'] * 3) / fast_move['energyGain']) - counts[0] - counts[1])

    return counts

def turn_alignment(my_fast,opp_fast):
    pass

def get_moveset_and_counts(pokemon_name, pokemon_data, move_data):
    moveset = None
    for pokemon in pokemon_data:
        if pokemon_name.lower() == pokemon['speciesName'].lower():
            moveset = pokemon['moveset']
            break

    if moveset is None:
        return None , 0

    fast_move = moveset[0]
    charged1 = moveset[1]
    charged2 = moveset[2]

    fast_move_data = None
    for move in move_data:
        if fast_move.lower() == move['moveId'].lower():
            fast_move_data = move
            break

    if fast_move_data is None:
        return None , 0

    fast_count = round(fast_move_data['cooldown'] / 500)
    move_counts = {}
    move_counts[fast_move] = fast_count
    for charged_move_name in [charged1, charged2]:
        for move in move_data:
            if charged_move_name.lower() == move['moveId'].lower():
                move_counts[charged_move_name] = calculate_move_counts(fast_move_data, move)
                break

    return move_counts , fast_count

def mse(image1, image2):
    if image1.size == 0 or image2.size == 0:
        return float("inf")
    error = np.sum((image1.astype("float") - image2.astype("float")) ** 2)
    error /= float(image1.shape[0] * image1.shape[1])
    return error


# In[ ]:


scrcpy_window,phone = find_scrcpy_window(phones)
roi_adjust = [[23,175,415],[15,150,390]]
roi_adjust2 =[[50,370,860],[50,350,860]]
roi_adjust = roi_adjust[phone]
roi_adjust2 = roi_adjust2[phone_t]
roi_dim = [530,50]
my_roi = (roi_adjust2[0], roi_adjust2[1], roi_dim[0], roi_dim[1])
opp_roi = (roi_adjust2[2], roi_adjust2[1], roi_dim[0], roi_dim[1])
title_bar_height = get_title_bar_height()
# Load the template images in color format
my_pokemon_template_color = cv2.imread('templates/my-temp2.png')
opp_pokemon_template_color = cv2.imread('templates/opp-temp2.png')
my_pokemon_template_color = cv2.resize(my_pokemon_template_color, (269, 77))
opp_pokemon_template_color = cv2.resize(opp_pokemon_template_color, (269, 77))
# Convert the template images to grayscale
my_pokemon_template = cv2.cvtColor(my_pokemon_template_color, cv2.COLOR_BGR2GRAY)
opp_pokemon_template = cv2.cvtColor(opp_pokemon_template_color, cv2.COLOR_BGR2GRAY)


# In[ ]:


prev_my_roi_img = np.array([])
prev_opp_roi_img = np.array([])
prev_corrected_my_name = None
prev_corrected_opp_name = None
threshold = 500 
print_out = False
display_img = True
update_timer = 500
league = None
def update_ui():
    global my_pokemon_label, opp_pokemon_label, my_moveset_label, opp_moveset_label, screenshot_label, correct_alignment
    global prev_my_roi_img, prev_opp_roi_img,corrected_my_name , corrected_opp_name, my_fast_move_turns, opp_fast_move_turns
    global prev_corrected_my_name, prev_corrected_opp_name, league, league_pok

    screen = client.last_frame
    time_start = time.time()
    if screen is not None:
        my_roi_img = screen[my_roi[1]:my_roi[1] + my_roi[3], my_roi[0]:my_roi[0] + my_roi[2]]
        opp_roi_img = screen[opp_roi[1]:opp_roi[1] + opp_roi[3], opp_roi[0]:opp_roi[0] + opp_roi[2]]

        if mse(my_roi_img, prev_my_roi_img) > threshold or mse(opp_roi_img, prev_opp_roi_img) > threshold:
            prev_my_roi_img = my_roi_img.copy()
            prev_opp_roi_img = opp_roi_img.copy()
            gray_my_roi = cv2.cvtColor(my_roi_img, cv2.COLOR_BGR2GRAY)
            gray_opp_roi = cv2.cvtColor(opp_roi_img, cv2.COLOR_BGR2GRAY)

            # Apply Gaussian blur to the grayscale images
            blur_my_roi = cv2.GaussianBlur(gray_my_roi, (5, 5), 0)
            blur_opp_roi = cv2.GaussianBlur(gray_opp_roi, (5, 5), 0)

            # Apply binary thresholding
            _, thresh_my_roi = cv2.threshold(blur_my_roi, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            _, thresh_opp_roi = cv2.threshold(blur_opp_roi, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            # Apply morphological operations to remove noise
            kernel = np.ones((2, 2), np.uint8)
            thresh_my_roi = cv2.morphologyEx(thresh_my_roi, cv2.MORPH_OPEN, kernel)
            thresh_opp_roi = cv2.morphologyEx(thresh_opp_roi, cv2.MORPH_OPEN, kernel)

            thresh_my_roi = Image.fromarray(thresh_my_roi)
            thresh_opp_roi = Image.fromarray(thresh_opp_roi)
            with PyTessBaseAPI(psm=PSM.AUTO_OSD) as api:
                api.SetImage(thresh_my_roi)
                api.Recognize()
                my_info = api.GetUTF8Text()

                api.SetImage(thresh_opp_roi)
                api.Recognize()
                opp_info = api.GetUTF8Text()

            # Display the extracted Pokémon name and CP value
            if print_out:
                print("My Info:", my_info)
                print("Opponent Info:", opp_info)

            if league is None:
                # Extract CP values using regex
                my_cp = re.search(r'\bCP\s+(\d+)\b', my_info)
                opp_cp = re.search(r'\bCP\s+(\d+)\b', opp_info)
                if my_cp:
                    my_cp = int(my_cp.group(1))
                    print(f"My Pokémon CP: {my_cp}")
                if opp_cp:
                    opp_cp = int(opp_cp.group(1))
                    print(f"Opponent Pokémon CP: {opp_cp}")
                if my_cp and opp_cp:
                    higher_cp = max(my_cp, opp_cp)
                    if higher_cp <= 1500:
                        league = "great-league"
                    elif 1500 < higher_cp <= 2500:
                        league = "ultra-league"
                    else:
                        league = "master-league"

                    print(f"League: {league}")
                    league_pok = f"json_files/{league.lower()}.json"
                    try:
                        with open(league_pok, 'r') as file:
                            league_pok = json.load(file)
                            print(f"Loaded {league} JSON data")
                    except FileNotFoundError:
                        print(f"Failed to load {league} JSON data")
                else:
                    print("Could not determine league")    

            # Extract Pokémon names using regex
            my_info_match = re.search(r'([A-Z][a-z]+(?:\s[A-Z][a-z]+)?)', my_info)
            opp_info_match = re.search(r'([A-Z][a-z]+(?:\s[A-Z][a-z]+)?)', opp_info)

            if my_info_match and opp_info_match and league is not None:

                my_info_name = my_info_match.group(0)
                opp_info_name = opp_info_match.group(0)

                temp_corrected_my_name = closest_pokemon_name(my_info_name, pokemon_names)
                temp_corrected_opp_name = closest_pokemon_name(opp_info_name, pokemon_names)

                # Check if either Pokémon is Giratina and change form based on the league
                if temp_corrected_my_name == "Giratina":
                    if league in ["great-league", "ultra-league"]:
                        temp_corrected_my_name = "Giratina (Altered)"
                    else:
                        temp_corrected_my_name = "Giratina (Origin)"

                if temp_corrected_opp_name == "Giratina":
                    if league in ["great-league", "ultra-league"]:
                        temp_corrected_opp_name = "Giratina (Altered)"
                    else:
                        temp_corrected_opp_name = "Giratina (Origin)"

                if temp_corrected_my_name:
                    corrected_my_name = temp_corrected_my_name
                    prev_corrected_my_name = corrected_my_name
                else:
                    corrected_my_name = prev_corrected_my_name
                    if print_out:
                        print(f"Using previous Pokémon name for My Pokémon: {corrected_my_name}")

                if temp_corrected_opp_name:
                    corrected_opp_name = temp_corrected_opp_name
                    prev_corrected_opp_name = corrected_opp_name
                else:
                    corrected_opp_name = prev_corrected_opp_name
                    if print_out:
                        print(f"Using previous Pokémon name for Opponent Pokémon: {corrected_opp_name}")

            else:
                corrected_my_name = prev_corrected_my_name
                corrected_opp_name = prev_corrected_opp_name
                if print_out:
                    print("Could not extract Pokémon names from the OCR output.")
                    print(f"Using previous Pokémon name for My Pokémon: {corrected_my_name}")
                    print(f"Using previous Pokémon name for Opponent Pokémon: {corrected_opp_name}")

            if corrected_opp_name:
                opp_move_counts, opp_fast_move_turns = get_moveset_and_counts(corrected_opp_name, league_pok, moves)
                if opp_move_counts is not None:
                    opp_moveset_text = ', '.join([f"{move.lower()}: {count}" for move, count in opp_move_counts.items()])
                    opp_moveset_label.setText(f"Opponent Moveset: {opp_moveset_text}")
                    if print_out:
                        print(f"Opponent Pokémon: {corrected_opp_name}")
                        print(opp_move_counts)
                else:
                    opp_moveset_label.setText("Opponent Moveset: Error")
                    if print_out:
                        print("Error getting opponent moveset.")

            if corrected_my_name:
                my_move_counts, my_fast_move_turns = get_moveset_and_counts(corrected_my_name, league_pok, moves)
                if my_move_counts is not None:
                    my_moveset_text = ', '.join([f"{move.lower()}: {count}" for move, count in my_move_counts.items()])
                    my_moveset_label.setText(f"My Moveset: {my_moveset_text}")
                    if print_out:
                        print(f"My Pokémon: {corrected_my_name}")
                        print(my_move_counts)
                else:
                    my_moveset_label.setText("My Moveset: Error")
                    if print_out:
                        print("Error getting my moveset.")

        else:
            if print_out:
                print('No change')

        # Draw rectangles around the ROI
        roi_color = (0, 255, 0)  
        screen_with_rois = cv2.rectangle(screen.copy(), (my_roi[0], my_roi[1]), (my_roi[0] + my_roi[2], my_roi[1] + my_roi[3]), roi_color, 2)
        screen_with_rois = cv2.rectangle(screen_with_rois, (opp_roi[0], opp_roi[1]), (opp_roi[0] + opp_roi[2], opp_roi[1] + opp_roi[3]), roi_color, 2)

        # Update the UI labels
        my_pokemon_label.setText(f"My Pokémon: {corrected_my_name}")
        opp_pokemon_label.setText(f"Opponent Pokémon: {corrected_opp_name}")

        if corrected_my_name is None:
            opp_moveset_label.setText(f"Opponent Moveset: {None}")
        if corrected_opp_name is None:
            my_moveset_label.setText(f"My Moveset: {None}")

        if corrected_my_name is not None and corrected_opp_name is not None:
            my_count = my_fast_move_turns
            opp_count = opp_fast_move_turns
            try:
                correct_count = df.loc[my_count, str(opp_count)]
            except KeyError:
                correct_count = "Unknown"
            correct_alignment.setText(f"Correct Alignemnt: {correct_count}")
        # Resize the image and fix the colors

        if display_img:
            scale_percent = 10
            width = int(screen_with_rois.shape[1] * scale_percent / 100)
            height = int(screen_with_rois.shape[0] * scale_percent / 100)
            dim = (width, height)
            resized_image = cv2.resize(screen_with_rois, dim, interpolation=cv2.INTER_AREA)
            resized_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)

            # clear_output(wait=True)
            # Update the screenshot display
            height, width, channel = resized_image.shape
            bytes_per_line = channel * width
            qimage = QImage(resized_image.data, width, height, bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(qimage)
            screenshot_label.setPixmap(pixmap)

    time_elapsed = time.time() - time_start
    timer_label.setText(f"elapsed Time: {round(time_elapsed,2):0.3f}")
    # Schedule the next update
    QTimer.singleShot(update_timer, update_ui)


app = QApplication(sys.argv)
window = QWidget()
window.setWindowTitle("Pokémon Information Display")
layout = QVBoxLayout()

opp_pokemon_label = QLabel("Opponent Pokémon:")
layout.addWidget(opp_pokemon_label)

opp_moveset_label = QLabel()
layout.addWidget(opp_moveset_label)

correct_alignment = QLabel()
layout.addWidget(correct_alignment)

my_pokemon_label = QLabel("My Pokémon:")
layout.addWidget(my_pokemon_label)

my_moveset_label = QLabel()
layout.addWidget(my_moveset_label)

timer_label = QLabel()
layout.addWidget(timer_label, alignment=Qt.AlignRight) 

if display_img:
    screenshot_label = QLabel()
    layout.addWidget(screenshot_label)

# Add the Exit button and its signal connection
# exit_button = QPushButton("Exit")
# exit_button.clicked.connect(app.quit)
# layout.addWidget(exit_button)

window.setLayout(layout)
window.show()

# Start the update loop
QTimer.singleShot(update_timer, update_ui)

app.exec_()

