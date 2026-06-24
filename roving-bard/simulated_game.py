# Copyright 2026 Google LLC
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import tkinter as tk


class SimulatedGame:
    def __init__(self, root):
        self.root = root
        self.root.title("Simulated Game Client")
        self.root.geometry("800x600+100+100")  # Fixed position and size

        # Game regions mapping to display values
        self.regions = {
            "Town": {
                "name": "Town Center",
                "coords": "19.3N, 70.9W",
                "bg": "#4a7a4a",  # Forest green
            },
            "Forest": {
                "name": "Deep Forest",
                "coords": "14.9S, 103.1E",
                "bg": "#1e3d1e",  # Dark green
            },
            "Boss Arena": {
                "name": "Boss Arena",
                "coords": "42.0N, 12.5E",
                "bg": "#4d1a1a",  # Dark red
            },
            "Cave": {
                "name": "Secret Cave",
                "coords": "15.5N, 72.3W",  # Within cave coordinate range
                "bg": "#2b2b2b",  # Dark gray
            },
            "Unknown": {
                "name": "Wastelands",
                "coords": "0.0N, 0.0W",
                "bg": "#000000",  # Black
            },
        }

        # Init state
        self.current_region = "Town"

        # Main container with changing background
        self.game_viewport = tk.Frame(
            self.root, bg=self.regions[self.current_region]["bg"]
        )
        self.game_viewport.pack(fill=tk.BOTH, expand=True)

        # Label simulating main gameplay
        self.gameplay_label = tk.Label(
            self.game_viewport,
            text="[SIMULATED GAMEPLAY SCREEN]",
            fg="white",
            bg=self.regions[self.current_region]["bg"],
            font=("Helvetica", 16, "bold"),
        )
        self.gameplay_label.pack(pady=100)

        # Mini-map Widget (Top-Right of screen)
        # Position needs to align with the relative bounds in mapping.yaml:
        # x: 0.8 (80%), y: 0.05 (5%), w: 0.15 (15%), h: 0.15 (15%)
        # In an 800x600 window, this is top-right corner.
        self.minimap_frame = tk.Frame(self.root, bg="#111111", bd=2, relief=tk.RIDGE)
        # Place it absolutely to guarantee it matches bounds mapping
        self.minimap_frame.place(relx=0.8, rely=0.05, relwidth=0.15, relheight=0.15)

        # Mini-map title/graphic simulation
        self.minimap_graphic = tk.Label(
            self.minimap_frame,
            text="[MAP]",
            fg="cyan",
            bg="#111111",
            font=("Courier", 10, "bold"),
        )
        self.minimap_graphic.pack(pady=2)

        # Location Label inside/below minimap
        self.loc_label = tk.Label(
            self.minimap_frame,
            text=self.regions[self.current_region]["name"],
            fg="#ffffff",
            bg="#111111",
            font=("Helvetica", 9, "bold"),
        )
        self.loc_label.pack()

        # Coordinates Label inside/below minimap
        self.coord_label = tk.Label(
            self.minimap_frame,
            text=self.regions[self.current_region]["coords"],
            fg="#aaaaaa",
            bg="#111111",
            font=("Courier", 8),
        )
        self.coord_label.pack()

        # Control Panel (Bottom) for triggering region transitions
        self.control_panel = tk.Frame(self.root, bg="#222", height=100)
        self.control_panel.pack(fill=tk.X, side=tk.BOTTOM)

        tk.Label(
            self.control_panel,
            text="Region Controls:",
            fg="white",
            bg="#222",
            font=("Helvetica", 10, "bold"),
        ).pack(side=tk.LEFT, padx=10)

        for reg_name in self.regions.keys():
            btn = tk.Button(
                self.control_panel,
                text=reg_name,
                command=lambda r=reg_name: self.change_region(r),
                bg="#333",
                fg="white",
                activebackground="#555",
                activeforeground="white",
            )
            btn.pack(side=tk.LEFT, padx=5, pady=10)

    def change_region(self, region_name):
        self.current_region = region_name
        reg_data = self.regions[region_name]

        # Update view
        self.game_viewport.config(bg=reg_data["bg"])
        self.gameplay_label.config(
            bg=reg_data["bg"],
            text=f"[SIMULATED GAMEPLAY SCREEN: {region_name.upper()}]",
        )
        self.loc_label.config(text=reg_data["name"])
        self.coord_label.config(text=reg_data["coords"])
        print(f"[Game Client] Player moved to {region_name} ({reg_data['coords']})")


def main():
    root = tk.Tk()
    _ = SimulatedGame(root)
    root.mainloop()


if __name__ == "__main__":
    main()
