# Pokémon GO PVP Battle Assistant

This repository contains a Python script that assists players during Pokémon GO PVP battles. The script uses Pokémon and league-specific information from JSON files to display the move-set for each Pokémon, based on the recommended move-set from the PvPoke database. It also provides move counts and recommendations for move timings to help players avoid throwing moves on alignment.

# Features
* Automatic league detection (Great, Ultra, or Master) based on the Pokémon's CP
* Displays the move-set suggested based on PvPoke recommendations and their counts
* Calculates the best move timing to avoid throwing on alignment and giving opponent free turns

# To-Do
* Add memory and display all opponent Pokémon seen so far
* Implement CV/ML (YOLO) to detect Pokemon and actual move-sets from in-game animations
* Keep track of actual energy each Pokemon has during battle


# Installation
1. Clone the repo: 
```
git clone git@github.com:basemprince/pogo-bot.git
```
2. Navigate to the cloned repository and install the required Python packages:
```
cd pogo-bot
pip install -r requirements.txt
```
3. Run the main script:
```
python main.py
```
Now, the pogo-bot should show up in a new window as shown:
![App UI Screenshot](templates/app-ui.png "App UI")

