import sys
import tkinter as tk
from tkinter import scrolledtext
from double_positive import DoublePositive

class LogRedirector:
    def __init__(self, widget):
        self.widget = widget

    def write(self, text):
        text = text.replace('\r', '\n')
        self.widget.config(state=tk.NORMAL)
        self.widget.insert(tk.END, text)
        self.widget.see(tk.END)
        self.widget.config(state=tk.DISABLED)
        
        if '\n' in text:self.flush()

    def flush(self):
        self.widget.update_idletasks()


class TkinterWrapper:
    def __init__(self, root):
        self.root = root
        self.root.title("Double Positive Analysis")
        self.root.geometry("1080x720")
        self.root.resizable(True, True)
        self.HEADER_FONT = ("Arial", 20, "bold")
        self.BODY_FONT = ("Arial", 20)
        self.LOG_FONT = ("Courier", 15)
        self.ENTRY_WIDTH = 80
    
    def create_widgets(self):
        # --- Main Frame ---
        self.main_frame = tk.Frame(self.root, padx=15, pady=15)
        self.main_frame.pack(fill='both', expand=True)

        # --- Input/Button Frame (at the top) ---
        self.login_frame = tk.Frame(self.main_frame)
        self.login_frame.pack(pady=10)

        
        self.image_path_label = tk.Label(self.login_frame, text="Image Path:", font=self.HEADER_FONT)
        self.image_path_label.grid(row=0, column=0, padx=5, pady=5, sticky="w")
        self.image_path_entry = tk.Entry(self.login_frame, width=self.ENTRY_WIDTH, font=self.BODY_FONT)
        self.image_path_entry.grid(row=0, column=1, padx=5, pady=5)

        self.positive_threshold_label = tk.Label(self.login_frame, text="Positive Threshold:", font=self.HEADER_FONT)
        self.positive_threshold_label.grid(row=1, column=0, padx=5, pady=5, sticky="w")
        self.positive_threshold_entry = tk.Entry(self.login_frame, width=self.ENTRY_WIDTH, font=self.BODY_FONT)
        self.positive_threshold_entry.grid(row=1, column=1, padx=5, pady=5)
        self.positive_threshold_entry.insert(0, "0.05")
        
        self.cluster_threshold_label = tk.Label(self.login_frame, text="Cluster Threshold:", font=self.HEADER_FONT)
        self.cluster_threshold_label.grid(row=2, column=0, padx=5, pady=5, sticky="w")
        self.cluster_threshold_entry = tk.Entry(self.login_frame, width=self.ENTRY_WIDTH, font=self.BODY_FONT)
        self.cluster_threshold_entry.grid(row=2, column=1, padx=5, pady=5)
        self.cluster_threshold_entry.insert(0, "20")
        
        self.remember_var = tk.IntVar(value=0)
        self.switch_label = tk.Label(self.login_frame, text="Show Animation", font=self.HEADER_FONT)
        self.switch_label.grid(row=3, column=0, padx=5, pady=10, sticky="w")

        # Sub-frame to hold the radio buttons side-by-side
        self.switch_frame = tk.Frame(self.login_frame)
        self.switch_frame.grid(row=3, column=1, padx=5, pady=10, sticky="w")
        # YES Radiobutton (Value=1)
        yes_radio = tk.Radiobutton(self.switch_frame, text="Yes", variable=self.remember_var, value=1, font=self.BODY_FONT,padx=10, pady=5)
        yes_radio.pack(side=tk.LEFT, padx=10)

        # NO Radiobutton (Value=0, default)
        no_radio = tk.Radiobutton(self.switch_frame, text="No", variable=self.remember_var, value=0, font=self.BODY_FONT,padx=10, pady=5)
        no_radio.pack(side=tk.LEFT, padx=10)

        self.process_button = tk.Button(self.login_frame, text="Process", command=self.trigger, width=25, bg="#4CAF50", fg="white")
        self.process_button.grid(row=4, column=0, columnspan=2, pady=15)
    
    def create_log_area(self):
        # --- Log Area Label ---
        self.log_label = tk.Label(self.main_frame, text="--- Program Log Output ---", font=self.HEADER_FONT)
        self.log_label.pack(pady=(5, 5))

        # --- ScrolledText Widget (The new element!) ---
        # This widget is a combination of a Text widget and a Scrollbar.
        self.log_area = scrolledtext.ScrolledText(self.main_frame, wrap=tk.WORD, width=45, height=15, font=self.LOG_FONT)
        self.log_area.pack(padx=10, pady=5, fill='both', expand=True)

        # Configure tags for styling output messages (optional but helpful)
        self.log_area.tag_config("error", foreground="red", font=("Courier", 9, "bold"))
        self.log_area.tag_config("success", foreground="green", font=("Courier", 9, "bold"))

        # Initially disable editing by the user
        self.log_area.config(state=tk.DISABLED)
        
        # Redirect standard output and error to the log area
        self.log_redirector = LogRedirector(self.log_area)
        sys.stdout = self.log_redirector
        sys.stderr = self.log_redirector
    
    def trigger(self):
        self.log_area.config(state=tk.NORMAL) 
        self.log_area.delete('1.0', tk.END)
        self.log_area.config(state=tk.DISABLED)
        
        image_path = self.image_path_entry.get()
        positive_threshold = self.positive_threshold_entry.get()
        cluster_threshold = self.cluster_threshold_entry.get()
        show_animation = bool(self.remember_var.get())
        dp = DoublePositive(path=image_path, positive_threshold=float(positive_threshold), cluster_threshold=int(cluster_threshold))
        self.log_message(f"--- Trigger Initiated with Image Path: {image_path}\n, Positive Threshold: {positive_threshold}\n, Cluster Threshold: {cluster_threshold}\n ---")
        dp.process()
        dp.save_statistics()
        if show_animation: dp.plot_3d()

    
    def log_message(self, message, tag=None):
        """Inserts a message into the scrollable log area."""
        # Ensure the Text widget is editable temporarily
        self.log_area.config(state=tk.NORMAL)
        
        # Insert the message at the end ('end')
        self.log_area.insert(tk.END, f"{message}\n", tag)
        
        # Scroll to the bottom to show the newest message
        self.log_area.see(tk.END)
        
        # Make the Text widget read-only again
        self.log_area.config(state=tk.DISABLED)
    
    def run(self):
        self.create_widgets()
        self.create_log_area()
        self.log_message("System ready. Waiting for user input.")
        self.root.mainloop()

if __name__ == "__main__":
    root = tk.Tk()
    app = TkinterWrapper(root)
    app.run()
