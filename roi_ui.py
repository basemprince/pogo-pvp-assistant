import cv2
import scrcpy.core as scrcpy
import tkinter as tk
import customtkinter as ctk
import utils
from PIL import Image, ImageTk
import yaml
import os


class DraggableResizableRectangle:
    def __init__(self, canvas, x1, y1, x2, y2,handle_color='red', handle_size=6):
        self.canvas = canvas

        self.rect_id = canvas.create_rectangle(x1, y1, x2, y2,outline=handle_color, width=2)
        self.handles = {}
        self.handle_size = handle_size
        self.handle_color = handle_color
        self.cursors = {'nw': 'cross', 'n': 'sb_v_double_arrow', 'ne': 'cross',
                        'e': 'sb_h_double_arrow', 'se': 'cross', 's': 'sb_v_double_arrow',
                        'sw': 'cross', 'w': 'sb_h_double_arrow'}

        for loc in ('nw', 'n', 'ne', 'e', 'se', 's', 'sw', 'w'):
            self.handles[loc] = canvas.create_rectangle(0, 0, handle_size, handle_size, fill=handle_color)
        self.update_handles()

        canvas.tag_bind(self.rect_id, '<Button-1>', self.on_start)
        canvas.tag_bind(self.rect_id, '<B1-Motion>', self.on_drag)
        canvas.tag_bind(self.rect_id, '<ButtonRelease-1>', self.on_drop)

        for handle in self.handles:
            canvas.tag_bind(self.handles[handle], '<Button-1>', self.on_start_resize)
            canvas.tag_bind(self.handles[handle], '<B1-Motion>', self.on_resize)
            canvas.tag_bind(self.handles[handle], '<ButtonRelease-1>', self.on_drop)
            canvas.tag_bind(self.handles[handle], '<Enter>', lambda e, h=handle: self.change_cursor(h))
            canvas.tag_bind(self.handles[handle], '<Leave>', lambda e, h=handle: self.reset_cursor(h))

    def change_cursor(self, handle):
        self.canvas.config(cursor=self.cursors[handle])

    def reset_cursor(self, handle):
        self.canvas.config(cursor='')


    def update_handles(self):
        cx1, cy1, cx2, cy2 = self.canvas.coords(self.rect_id)
        dx = self.handle_size / 2

        self.canvas.coords(self.handles['nw'], cx1 - dx, cy1 - dx, cx1 + dx, cy1 + dx)
        self.canvas.coords(self.handles['n'], (cx1 + cx2) / 2 - dx, cy1 - dx, (cx1 + cx2) / 2 + dx, cy1 + dx)
        self.canvas.coords(self.handles['ne'], cx2 - dx, cy1 - dx, cx2 + dx, cy1 + dx)
        self.canvas.coords(self.handles['e'], cx2 - dx, (cy1 + cy2) / 2 - dx, cx2 + dx, (cy1 + cy2) / 2 + dx)
        self.canvas.coords(self.handles['se'], cx2 - dx, cy2 - dx, cx2 + dx, cy2 + dx)
        self.canvas.coords(self.handles['s'], (cx1 + cx2) / 2 - dx, cy2 - dx, (cx1 + cx2) / 2 + dx, cy2 + dx)
        self.canvas.coords(self.handles['sw'], cx1 - dx, cy2 - dx, cx1 + dx, cy2 + dx)
        self.canvas.coords(self.handles['w'], cx1 - dx, (cy1 + cy2) / 2 - dx, cx1 + dx, (cy1 + cy2) / 2 + dx)

    def on_start(self, event):
        self.start_x = event.x
        self.start_y = event.y

    def on_drag(self, event):
        dx = event.x - self.start_x
        dy = event.y - self.start_y
        self.canvas.move(self.rect_id, dx, dy)
        self.update_handles()
        self.start_x = event.x
        self.start_y = event.y

    def on_drop(self, event):
        pass 

    def on_start_resize(self, event):
        self.start_x = event.x
        self.start_y = event

    def on_start_resize(self, event):
        self.start_x = event.x
        self.start_y = event.y
        self.start_rect_coords = self.canvas.coords(self.rect_id)

    def on_resize(self, event):
        dx = event.x - self.start_x
        dy = event.y - self.start_y

        rect_coords = list(self.start_rect_coords) 

        # Update the rectangle coordinates based on which handle is being dragged
        for handle in self.handles:
            if self.canvas.find_withtag(tk.CURRENT)[0] == self.handles[handle]:
                if 'w' in handle:
                    rect_coords[0] += dx
                if 'n' in handle:
                    rect_coords[1] += dy
                if 'e' in handle:
                    rect_coords[2] += dx
                if 's' in handle:
                    rect_coords[3] += dy

        # Apply the new rectangle coordinates and update the handles
        self.canvas.coords(self.rect_id, *rect_coords)
        self.update_handles()

    def get_coords(self):
        return self.canvas.coords(self.rect_id)

class RoiSelector(ctk.CTk):
    def __init__(self, client,img_scale=0.3,update_timer=1):
        super().__init__()
        self.title("Roi Selector")
        self.img_scale = img_scale
        self.client = client
        self.feed_res = (int(self.client.resolution[0]*self.img_scale), int(self.client.resolution[1]*self.img_scale))
        self.original_res = self.client.resolution
        self.update_timer = update_timer
        w, h = self.feed_res

        mainframe = ctk.CTkFrame(self)
        mainframe.grid(column=0, row=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=0, pady=0)

        # Message for users
        message_label = tk.Label(mainframe, text="New phone detected. Please align the ROIs to the correct locations. Once done, click Save and close the window.")
        message_label.grid(column=0, row=0, pady=10)

        self.save_button = ctk.CTkButton(mainframe, text="Save", command=self.save_coords)
        self.save_button.grid(column=0, row=1, pady=10)

        self.canvas = tk.Canvas(mainframe, width=self.feed_res[0], height=self.feed_res[1])
        self.canvas.grid(column=0, row=3, pady=0)

        starting_loc_my = [int(w * 0.07), int(h * 0.13), int(w * 0.5), int(h * 0.15)]
        self.my_rect = DraggableResizableRectangle(self.canvas, *starting_loc_my)


        starting_loc_msgs = [int(w * 0.25), int(h * 0.39), int(w * 0.9), int(h * 0.43)]
        self.msgs_rect = DraggableResizableRectangle(self.canvas, *starting_loc_msgs,handle_color='orange')

        starting_loc_pokeballs = [int(w * 0.07), int(h * 0.15), int(w * 0.22), int(h * 0.18)]
        self.my_pokeballs_rect = DraggableResizableRectangle(self.canvas, *starting_loc_pokeballs,handle_color='blue')

        starting_loc_typing = [int(w * 0.02), int(h * 0.1), int(w * 0.13), int(h * 0.13)]
        self.my_typing_rect = DraggableResizableRectangle(self.canvas, *starting_loc_typing,handle_color='green')

        starting_loc_charge_mv = [int(w * 0.35), int(h * 0.85), int(w * 0.7), int(h * 0.99)]
        self.first_charge_mv_rect = DraggableResizableRectangle(self.canvas, *starting_loc_charge_mv,handle_color='black')


        self.screen = None

        self.tk_img = ImageTk.PhotoImage(Image.new("RGB", self.feed_res))
        self.image_on_canvas = self.canvas.create_image(0, 0, image=self.tk_img, anchor=tk.NW)


    def save_coords(self):
        def get_scaled_coords(rect):
            x1, y1, x2, y2 = rect.get_coords()
            width = x2 - x1
            height = y2 - y1

            # Offset adjustment
            x_offset = 0 
            y_offset = 0
            x1 += x_offset * self.img_scale
            y1 += y_offset * self.img_scale

            roi_coords = []
            for i, coord in enumerate([x1, y1, width, height]):
                original_coord = int(coord / (self.feed_res[i % 2] / self.original_res[i % 2]))
                roi_coords.append(original_coord)

            return roi_coords

        def get_mirror_coords(roi_coords):
            phone_width = self.original_res[0]
            opp_x = phone_width - roi_coords[0] - roi_coords[2]
            return [opp_x, roi_coords[1], roi_coords[2], roi_coords[3]]

        my_roi_coords = get_scaled_coords(self.my_rect)
        msgs_roi_coords = get_scaled_coords(self.msgs_rect)
        pokeballs_roi_coords = get_scaled_coords(self.my_pokeballs_rect)
        typing_roi_coords = get_scaled_coords(self.my_typing_rect)
        first_charge_mv_roi_coords = get_scaled_coords(self.first_charge_mv_rect)

        print(my_roi_coords,msgs_roi_coords,pokeballs_roi_coords,typing_roi_coords,first_charge_mv_roi_coords)

        opp_roi_coords = get_mirror_coords(my_roi_coords)
        opp_pokeballs_roi_coords = get_mirror_coords(pokeballs_roi_coords)
        opp_typing_roi_coords = get_mirror_coords(typing_roi_coords)
        opp_typing_roi_coords[0] -= 5
        opp_typing_roi_coords[1] -= 5

        second_mv_roi_coords = get_mirror_coords(first_charge_mv_roi_coords)

        phone_data = {
            'my_roi': my_roi_coords,
            'opp_roi': opp_roi_coords,  
            'msgs_roi': msgs_roi_coords,
            'my_pokeballs_roi': pokeballs_roi_coords,
            'opp_pokeballs_roi': opp_pokeballs_roi_coords,
            'my_typing_roi': typing_roi_coords,
            'opp_typing_roi': opp_typing_roi_coords,
            'first_charge_mv_roi': first_charge_mv_roi_coords,
            'second_charge_mv_roi': second_mv_roi_coords,
        }

        yaml_file = 'phone_roi.yaml'
        if os.path.exists(yaml_file):
            with open(yaml_file, 'r') as file:
                data = yaml.safe_load(file)
                if data is None:
                    data = {}
        else:
            data = {}

        phone_model = self.client.device_name
        if phone_model in data:
            data[phone_model].update(phone_data)
        else:
            data[phone_model] = phone_data

        with open(yaml_file, 'w') as file:
            yaml.dump(data, file)

    def update_ui(self, client):
        original_screen = client.last_frame
        if original_screen is not None:
            resized_image = cv2.resize(original_screen, self.feed_res, interpolation=cv2.INTER_AREA)
            pil_img  = Image.fromarray(resized_image)
            self.tk_img = ImageTk.PhotoImage(pil_img)

            # Update the image of the existing image item on the canvas
            self.canvas.itemconfig(self.image_on_canvas, image=self.tk_img)

            # Raise the rectangles and their handles to the top
            self.canvas.tag_raise(self.my_rect.rect_id)
            self.canvas.tag_raise(self.msgs_rect.rect_id)
            self.canvas.tag_raise(self.my_pokeballs_rect.rect_id)
            self.canvas.tag_raise(self.my_typing_rect.rect_id)
            self.canvas.tag_raise(self.first_charge_mv_rect.rect_id)
            for rect in [self.my_rect, self.msgs_rect, self.my_pokeballs_rect, self.my_typing_rect,self.first_charge_mv_rect]:
                for handle in rect.handles.values():
                    self.canvas.tag_raise(handle)

        self.after(self.update_timer, lambda: self.update_ui(client))


if __name__ == "__main__":
    img_scale = 0.5
    update_timer = 1
    client = utils.connect_to_device("127.0.0.1:5037")
    print(f"Connected to device with resolution: {client.resolution}")
    app = RoiSelector(client,img_scale,update_timer)
    app.update_ui(client)
    app.mainloop()