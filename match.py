import json
import math
import time
import utils
class Move:
    def __init__(self, name,move_id, move_type, category, energy):
        self.name = name
        self.move_id = move_id
        self.move_type = move_type  # move's type (Ghost, Steel, Grass, etc.)
        self.category = category  # Fast or Charge
        self.energy = energy  # For fast moves, it's energy gain. For charge moves, it's energy cost.

    def __repr__(self):
        return f"{self.__class__.__name__}(name={self.name}, move_type={self.move_type}, category={self.category}, energy={self.energy})"


class FastMove(Move):
    def __init__(self, name, move_id, move_type, energy_gain, cooldown):
        super().__init__(name, move_id, move_type, "fast", energy_gain)
        self.cooldown = cooldown
        self.move_turns = int(self.cooldown/500)
    def __repr__(self):
        return f"{self.__class__.__name__}(name={self.name}, move_type={self.move_type}, category={self.category}, energy={self.energy}, cooldown={self.cooldown}, move_turns={self.move_turns})"
    def move_count_str(self):
        return f'{self.name} - {self.move_turns}'

class ChargeMove(Move):
    def __init__(self, name, move_id, move_type, energy_cost, counts):
        super().__init__(name, move_id, move_type, "charge", energy_cost)
        self.counts = counts
    def __repr__(self):
        return f"{self.__class__.__name__}(name={self.name}, move_type={self.move_type}, category={self.category}, energy={self.energy}, counts={self.counts})"
    def move_count_str(self,fast_move):
        return f'{self.name} - {self.counts[fast_move]}'


class Pokemon:
    def __init__(self, name,id):
        self.species_name = name
        self.species_id = id
        self.fast_moves = {} 
        self.charge_moves = {}  
        self.energy = 0 # accumulated energy
        self.time_on_field = 0
        self.last_update_time = time.time()  # to calculate the energy accumulation
        self.recommended_moveset = None  # recommended Fast and Charge Moves based on league
        self.ui_chosen_moveset = None
        self.rating = None

    def __repr__(self):
        return f"{self.__class__.__name__}(species_name={self.species_name}, energy={self.energy}, time_on_field={self.time_on_field}, recommended_moveset={self.recommended_moveset})"
    
    def add_fast_move(self, name,id, move_type, energy_gain, cooldown):
        fast_move = FastMove(name, id, move_type, energy_gain, cooldown)
        self.fast_moves[id] = fast_move 

    def add_charge_move(self, name,id, move_type, energy_cost, counts):
        charge_move = ChargeMove(name, id, move_type, energy_cost, counts)
        self.charge_moves[id] = charge_move 

    def set_recommended_moveset(self, league_details):
        for league_pokemon in league_details:
            if self.species_name.lower() == league_pokemon['speciesName'].lower():
                self.recommended_moveset = league_pokemon['moveset']
                self.rating = league_pokemon['rating']
                break
        self.ui_chosen_moveset = self.recommended_moveset

    def load_fast_move(self, move_details,fast_move_name):
        for move in move_details:
            if move['moveId'] == fast_move_name:
                self.add_fast_move(move['name'],move['moveId'], move['type'], move['energyGain'], move['cooldown'])

    def load_charge_move(self, move_details,charge_move_name):
        for move in move_details:
            if move['moveId'] == charge_move_name:
                self.add_charge_move(move['name'],move['moveId'], move['type'], move['energy'], {})
                self.calculate_charge_move_counts(self.charge_moves[move['moveId']])
    
    def update_energy(self):
        current_time = time.time()
        time_difference = current_time - self.last_update_time
        self.energy += self.calculate_energy_gain(time_difference)
        self.last_update_time = current_time

    def calculate_energy_gain(self, time_difference):
        # logic for energy gain over time
        pass

    def calculate_charge_move_counts(self,charge_move):
        for fast_move_id, fast_move in self.fast_moves.items():
            self.charge_moves[charge_move.move_id].counts[fast_move_id] = self.charge_move_counts_helper(fast_move, charge_move)

    def charge_move_counts_helper(self,fast_move, charge_move):
        counts = []
        counts.append(math.ceil((charge_move.energy * 1) / fast_move.energy))
        counts.append(math.ceil((charge_move.energy * 2) / fast_move.energy) - counts[0])
        counts.append(math.ceil((charge_move.energy * 3) / fast_move.energy) - counts[0] - counts[1])

        return counts

def load_pk_data(info_name, pokemon_names, pokemon_details,moves_data,league_pok):
    temp_corrected_name = utils.closest_pokemon_name(info_name, pokemon_names)
    move_data = [item for item in pokemon_details if temp_corrected_name and item['speciesName'].startswith(temp_corrected_name)]
    
    pokemon_list = []
    
    for data in move_data:
        pokemon = Pokemon(data['speciesName'],data['speciesId'])
        
        for fast_move_name in data['fastMoves']:
            pokemon.load_fast_move(moves_data,fast_move_name)
        
        for charge_move_name in data['chargedMoves']:
            pokemon.load_charge_move(moves_data,charge_move_name)
        
        pokemon.set_recommended_moveset(league_pok)
        
        pokemon_list.append(pokemon)
    
    return pokemon_list


class Player:
    def __init__(self, name):
        self.name = name
        self.pokemons = [None, None, None]
        self.recommended_pk_ind = [None, None, None]
        self.current_pokemon_index = None  # Index of the current Pokemon on the field
        self.shield_count = 2  
        self.pokemon_count = 0
        self.switch_lock = False
        self.switch_lock_timer = 0 
        self.switch_out_time = None
        self.oldest_pokemon_index = 0  # Added to track the index of the oldest Pokemon

    def add_pokemon(self, pokemon_list):
        if not pokemon_list:
            return False  # Skip this list if it's empty
        pokemon_list = [pokemon for pokemon in pokemon_list if pokemon.recommended_moveset]  # Filter out Pokemon with empty recommended_moveset
        # Check if any Pokemon in the list is already in the pokemons list
        for i, slot in enumerate(self.pokemons):
            if slot and any(p.species_id == p2.species_id for p2 in slot for p in pokemon_list):
                self.update_current_pokemon_index(i)  # Update index to the existing Pokemon
                return False  # Skip if Pokemon already in list

        if self.pokemon_count < 3:
            self.pokemons[self.pokemon_count] = pokemon_list
            self.update_current_pokemon_index(self.pokemon_count)
            self.update_recommended_pk()
            self.pokemon_count += 1
        else:
            self.pokemons[self.oldest_pokemon_index] = pokemon_list  # Replace the oldest Pokemon with the new one
            self.update_current_pokemon_index(self.oldest_pokemon_index)  
            self.update_recommended_pk()
            self.oldest_pokemon_index = (self.oldest_pokemon_index + 1) % 3  # Update the index of the oldest Pokemon
        return True


    def update_recommended_pk(self):
        max_rating_index = None
        max_rating = -1  # Initialize to a low value

        for i, pk in enumerate(self.pokemons[self.current_pokemon_index]):
            if pk.rating is not None and pk.rating > max_rating:
                max_rating = pk.rating
                max_rating_index = i

        self.recommended_pk_ind[self.current_pokemon_index] = max_rating_index


    def update_current_pokemon_index(self, new_index):
        if new_index != self.current_pokemon_index:
            self.current_pokemon_index = new_index
            self.switch_out_time = time.time()
            self.switch_lock = True
            self.switch_lock_timer = 60
            self.countdown_switch_lock()
 
    def countdown_switch_lock(self):
        self.switch_lock_timer = 60 - int(time.time() - self.switch_out_time)
        if self.switch_lock_timer <= 0:
            self.switch_out_time = None
            self.switch_lock_timer = 0
            self.switch_lock = False

    def use_shield(self):
        if self.shield_count > 0:
            self.shield_count -= 1
        
    def ui_helper(self,pokemon_ind=None,chosen_pk_ind=None):
        pk_name = []
        pk_fast_moves = []
        pk_charge_moves = []

        if pokemon_ind is None:
            pokemon_ind = self.current_pokemon_index
            for pk in self.pokemons[pokemon_ind]:
                pk_name.append(pk.species_name)

        if chosen_pk_ind is None:
            pk_recom = self.pokemons[pokemon_ind][self.recommended_pk_ind[self.current_pokemon_index]]
        else:
            pk_recom = self.pokemons[pokemon_ind][chosen_pk_ind]
            
        for fast_mv in pk_recom.fast_moves.values():
            pk_fast_moves.append(fast_mv.move_count_str())
        for charge_mv in pk_recom.charge_moves.values():
            pk_charge_moves.append(charge_mv.move_count_str(pk_recom.recommended_moveset[0]))
            
        return pk_name, pk_fast_moves, pk_charge_moves

    def __repr__(self):
        return f"Player(name={self.name}, " \
               f"pokemons={[str(pokemon) for pokemon in self.pokemons]}, " \
               f"current_pokemon_index={self.current_pokemon_index}, " \
               f"pokemon_count={self.pokemon_count}, " \
               f"shield_count={self.shield_count}, " \
               f"switch_lock={self.switch_lock}, " \
               f"switch_lock_timer={self.switch_lock_timer})"

if __name__ == "__main__":
    pokemon_names = utils.load_pokemon_names()
    pokemon_details = utils.load_pokemon_details()
    moves = utils.load_moves_info()

    league_pok = f"json_files/rankings/Ultra League.json"
    with open(league_pok, 'r') as file:
        league_pok = json.load(file)

    pk = load_pk_data('tapu bulu',pokemon_names,pokemon_details,moves,league_pok)
    my_player = Player('me')
    my_player.add_pokemon(pk)
    print(my_player)