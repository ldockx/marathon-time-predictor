import tkinter as tk
from tkinter import ttk, messagebox
from src.utils.helper_functions import hms_to_hours

def get_user_input():
    def submit():
        try:
            values = {
                "total_km": float(km_entry.get()),
                "sp4week": float(sp_entry.get()),
                "hm_time": hm_entry.get(),
                "gender": gender_var.get(),
                "age": int(age_entry.get())
            }

            # Validate HM Time Format
            hms_to_hours(values["hm_time"])

            root.user_values = values
            root.destroy()

        except Exception as e:
            messagebox.showerror("Input Error", str(e))

    root = tk.Tk()
    root.title("Marathon Predictor — Input Data")

    frame = ttk.Frame(root, padding=25)
    frame.grid()

    # Styling title
    title = ttk.Label(frame, text="Enter Your Training Data", font=("Helvetica", 16, "bold"))
    title.grid(row=0, column=0, columnspan=2, pady=(0, 20))

    # Total KM
    ttk.Label(frame, text="Total KM last 4 weeks:").grid(row=1, column=0, sticky="w", pady=5)
    km_entry = ttk.Entry(frame, width=20)
    km_entry.grid(row=1, column=1, pady=5)

    # Speed
    ttk.Label(frame, text="Avg speed last 4 weeks (km/h):").grid(row=2, column=0, sticky="w", pady=5)
    sp_entry = ttk.Entry(frame, width=20)
    sp_entry.grid(row=2, column=1, pady=5)

    # Half Marathon Time
    ttk.Label(frame, text="Half Marathon Time (HH:MM:SS):").grid(row=3, column=0, sticky="w", pady=5)
    hm_entry = ttk.Entry(frame, width=20)
    hm_entry.grid(row=3, column=1, pady=5)

    # Gender
    ttk.Label(frame, text="Gender:").grid(row=4, column=0, sticky="w", pady=5)
    gender_var = tk.StringVar()
    gender_dropdown = ttk.Combobox(frame, textvariable=gender_var, values=["Male", "Female"], state="readonly")
    gender_dropdown.grid(row=4, column=1, pady=5)
    gender_dropdown.current(0)

    # Age
    ttk.Label(frame, text="Age:").grid(row=5, column=0, sticky="w", pady=5)
    age_entry = ttk.Entry(frame, width=20)
    age_entry.grid(row=5, column=1, pady=5)

    submit_button = ttk.Button(frame, text="Run Prediction", command=submit)
    submit_button.grid(row=6, column=0, columnspan=2, pady=20)

    root.mainloop()

    return getattr(root, "user_values", None)


def show_results(predictions):
    def run_again():
        results_window.destroy()

    def exit_app():
        results_window.destroy()
        raise SystemExit

    results_window = tk.Tk()
    results_window.title("Marathon Predictor — Results")

    # Outer Frame Background
    outer = tk.Frame(results_window, bg="#e8e8ff", padx=30, pady=30)
    outer.pack(fill="both", expand=True)

    # Title
    title = tk.Label(
        outer,
        text="Marathon Prediction Results",
        font=("Helvetica", 18, "bold"),
        bg="#e8e8ff",
        fg="#333"
    )
    title.pack(pady=(0, 20))

    # Inner white result box
    result_frame = tk.Frame(outer, bg="white", padx=20, pady=20, bd=2, relief="groove")
    result_frame.pack(pady=10)

    for model_name, pred_time in predictions.items():
        lbl = tk.Label(
            result_frame,
            text=f"{model_name}: {pred_time}",
            font=("Helvetica", 14),
            bg="white"
        )
        lbl.pack(anchor="w", pady=5)

    # Button row
    button_frame = tk.Frame(outer, bg="#e8e8ff")
    button_frame.pack(pady=20)

    run_again_btn = ttk.Button(button_frame, text="Run Again", command=run_again)
    run_again_btn.grid(row=0, column=0, padx=10)

    exit_btn = ttk.Button(button_frame, text="Exit Application", command=exit_app)
    exit_btn.grid(row=0, column=1, padx=10)

    results_window.mainloop()