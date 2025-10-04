# retinova_cli.py
"""
RetiNova CLI — robust terminal wrapper with strict interactive validation.

Interactive mode now:
 - asks for model path (loop until valid file)
 - shows menu with 3 options and re-prompts until a valid choice is entered:
     1) File path (loop until valid path)
     2) Paste base64/data URL (loop until valid base64)
     3) Read raw image bytes from stdin (use '-' as image arg when piping)
 - Also works with command-line args (model [image]) with validation loops when needed.

Saves:
 - gradcam_overlay.png
 - gradcam_heatmap.png
"""

import os
import sys
import warnings
import logging
import base64
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import cv2

# Silence TF logs
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore")
logging.getLogger("tensorflow").setLevel(logging.ERROR)
logging.getLogger("absl").setLevel(logging.ERROR)


class RetiNovaCLI:
    def __init__(self, model_path, img_size=(224, 224), conf_threshold=0.75):
        self.img_size = img_size
        self.conf_threshold = float(conf_threshold)

        self.conditions = [
            "AMD",
            "Diabetic Retinopathy",
            "Glaucoma",
            "Hypertensive Retinopathy",
            "Normal",
            "Optical Retinopathy",
            "RVO",
            "Retinal Tears/Detachments",
        ]

        self.choice_map = {0: "Low", 1: "Mild", 2: "Moderate", 3: "High"}
        self.choice_conf_map = {"Low": 0.0, "Mild": 0.05, "Moderate": 0.1, "High": 0.2}

        # load model defensively
        try:
            self.model = load_model(model_path, compile=False)
        except Exception as e:
            raise RuntimeError(f"Failed to load model from '{model_path}': {e}")

        # find last conv-like layer robustly
        self.last_conv_layer_name = None
        try:
            for layer in reversed(self.model.layers):
                lname = layer.__class__.__name__.lower()
                if "conv" in lname or "separableconv" in lname or "depthwiseconv" in lname:
                    self.last_conv_layer_name = layer.name
                    break
            if self.last_conv_layer_name is None:
                for layer in reversed(self.model.layers):
                    if 'conv' in layer.name.lower():
                        self.last_conv_layer_name = layer.name
                        break
        except Exception:
            self.last_conv_layer_name = None

        # full risk question bank (same as before)
        self.risk_questions = {
            "AMD":[
                ("Are you over 55 years old?", ["Under 55","55-64","65-74","75 or older"]),
                ("Do you have difficulty reading or recognizing faces?", ["No","Occasionally","Often","Very frequent/severe"]),
                ("Have you noticed straight lines appear wavy or distorted?", ["No","Rarely","Occasionally","Frequently/severe"]),
                ("Is your central vision blurry or do you see a blind spot?", ["No","Slight blurring","Moderate blurring","Severe/sudden blind spot"]),
                ("Is it harder to see in low light?", ["No difficulty","Slight difficulty","Noticeable difficulty","Severe difficulty"]),
                ("Do you have a family history of macular degeneration?", ["No","Distant relative","Parent/sibling","Multiple close family members"]),
                ("Do you smoke or have you smoked?", ["Never","Rarely/occasional","Moderate","Heavy/long-term"]),
                ("Any recent sudden worsening of central vision?", ["No","Slight change","Noticeable gradual change","Sudden severe"])
            ],
            "Diabetic Retinopathy":[
                ("How long have you had diabetes?", ["<5 years","5-10 years","10-20 years",">20 years"]),
                ("How well is your blood sugar controlled?", ["Excellent","Good","Fair","Poor"]),
                ("Any recent changes in vision?", ["No","Slight","Moderate","Severe"]),
                ("Do you have high blood pressure?", ["No","Occasionally","Often","Very frequent/severe"]),
                ("Any history of kidney disease or vascular problems?", ["No","Yes, minor","Yes, moderate","Yes, severe"]),
                ("Have you noticed difficulties seeing at night?", ["No","Slight","Moderate","Severe"]),
                ("Have you been treated for eye problems before?", ["No","Yes, minor","Yes, moderate","Yes, severe"]),
                ("Are you taking medications for diabetes or eye issues?", ["No","Occasionally","Often","Very frequent/severe"])
            ],
            "Glaucoma":[
                ("Do you have a family history of glaucoma?", ["No","Distant relative","Parent/sibling","Multiple close family members"]),
                ("Have you noticed gradual loss of peripheral vision?", ["No","Occasionally","Often","Very frequent/severe"]),
                ("Have you experienced eye pain, halos, or headaches?", ["No","Occasionally","Often","Very frequent/severe"]),
                ("Has your vision worsened or become patchy?", ["No","Slightly","Moderate","Severe"]),
                ("Are you over 40 or have other risk factors?", ["No","Yes"]),
                ("Have you had trauma or previous eye surgery?", ["No","Yes"]),
                ("Any recent changes in prescription glasses?", ["No","Yes"])
            ],
            "Hypertensive Retinopathy":[
                ("Do you have high blood pressure?", ["No","Occasionally","Often","Very frequent/severe"]),
                ("Duration of high blood pressure?", ["<5 years","5-10 years","10-20 years",">20 years"]),
                ("Symptoms like blurry vision/headaches?", ["No","Slight","Moderate","Severe"]),
                ("Any smoking, diabetes, cholesterol, heart/kidney disease?", ["No","Occasionally","Often","Very frequent/severe"]),
                ("Recent blood pressure spikes?", ["No","Occasionally","Often","Very frequent/severe"]),
                ("Eye changes during routine checkups?", ["No","Yes, minor","Yes, moderate","Yes, severe"]),
                ("Experience double vision or spots?", ["No","Occasionally","Often","Very frequent/severe"])
            ],
            "RVO":[
                ("Sudden painless vision loss or blurring?", ["No","Slight","Moderate","Severe"]),
                ("New floaters, dark spots, shadows?", ["No","Slight","Moderate","Severe"]),
                ("Diagnosed with hypertension, diabetes, heart disease?", ["No","Yes, minor","Yes, moderate","Yes, severe"]),
                ("History of stroke/clotting/high cholesterol?", ["No","Yes, minor","Yes, moderate","Yes, severe"]),
                ("Recent dizziness, weakness, neurological symptoms?", ["No","Slight","Moderate","Severe"]),
                ("Currently on blood-thinning/cardiovascular meds?", ["No","Occasionally","Often","Very frequent/severe"])
            ],
            "Retinal Tears/Detachments":[
                ("Sudden increase in floaters/flashes?", ["No","Slight","Moderate","Severe"]),
                ("Shadow or curtain affecting vision?", ["No","Slight","Moderate","Severe"]),
                ("Recent eye trauma or surgery?", ["No","Yes, minor","Yes, moderate","Yes, severe"]),
                ("Highly nearsighted?", ["No","Mild","Moderate","Severe"]),
                ("Distortion or blurring of vision?", ["No","Slight","Moderate","Severe"]),
                ("Family history of retinal detachment?", ["No","Distant relative","Parent/sibling","Multiple close family members"]),
                ("Diabetes or hypertension?", ["No","Yes, minor","Yes, moderate","Yes, severe"])
            ],
            "Optical Retinopathy":[
                ("Progressive vision loss or color issues?", ["No","Slight","Moderate","Severe"]),
                ("History of eye pain/neurological/systemic issues?", ["No","Slight","Moderate","Severe"]),
                ("Family history of vision loss or neurological disorders?", ["No","Distant relative","Parent/sibling","Multiple close family members"]),
                ("Recent infections, toxins, deficiencies?", ["No","Slight","Moderate","Severe"]),
                ("History of prior eye/brain surgery, trauma, or tumors?", ["No","Yes, minor","Yes, moderate","Yes, severe"])
            ],
            "Normal":[
                ("Any eye issues currently?", ["No","Slight","Moderate","Severe"])
            ]
        }

    # ---- Image loaders ----
    def preprocess_image_from_path(self, img_path):
        bgr = cv2.imread(img_path)
        if bgr is None:
            raise ValueError(f"Could not read image at {img_path}")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, self.img_size, interpolation=cv2.INTER_AREA)
        norm = resized.astype(np.float32) / 255.0
        batch = np.expand_dims(norm, axis=0)
        return batch, rgb.astype(np.uint8)

    def preprocess_image_from_bytes(self, img_bytes):
        nparr = np.frombuffer(img_bytes, np.uint8)
        bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if bgr is None:
            raise ValueError("Could not decode image bytes. Provide a valid image.")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, self.img_size, interpolation=cv2.INTER_AREA)
        norm = resized.astype(np.float32) / 255.0
        batch = np.expand_dims(norm, axis=0)
        return batch, rgb.astype(np.uint8)

    # ---- Prediction (defensive) ----
    def predict_condition(self, img_array):
        img_array = np.array(img_array, dtype=np.float32)
        if img_array.ndim == 3:
            img_array = np.expand_dims(img_array, axis=0)

        preds = self.model.predict(img_array, verbose=0)

        if isinstance(preds, (list, tuple)):
            preds = preds[-1]

        preds = np.array(preds)

        if preds.ndim == 1:
            preds = np.expand_dims(preds, axis=0)

        if preds.size == 1:
            idx = 0
            conf = float(preds.flatten()[0])
        else:
            idx = int(np.argmax(preds[0]))
            conf = float(preds[0, idx])

        condition = self.conditions[idx] if idx < len(self.conditions) else "Unknown"
        return condition, conf

    # ---- Grad-CAM++ (defensive) ----
    def make_gradcam_plus_plus(self, img_array, img_rgb, alpha=0.4):
        try:
            img_np = np.array(img_array, dtype=np.float32)
            if img_np.ndim == 3:
                img_np = np.expand_dims(img_np, axis=0)
        except Exception:
            print("Grad-CAM++: could not coerce input to numpy array; skipping heatmap.")
            return img_rgb, None

        if self.last_conv_layer_name is None:
            return img_rgb, None

        try:
            conv_layer = self.model.get_layer(self.last_conv_layer_name)
            grad_model = tf.keras.models.Model(inputs=self.model.input,
                                              outputs=[conv_layer.output, self.model.output])

            img_tensor = tf.convert_to_tensor(img_np, dtype=tf.float32)

            with tf.GradientTape(persistent=True) as tape2:
                with tf.GradientTape() as tape1:
                    conv_outputs, predictions = grad_model(img_tensor, training=False)

                    if isinstance(predictions, (list, tuple)):
                        predictions = predictions[-1]

                    predictions = tf.convert_to_tensor(predictions)
                    if predictions.shape.rank == 0:
                        predictions = tf.reshape(predictions, (1, 1))
                    elif predictions.shape.rank == 1:
                        predictions = tf.expand_dims(predictions, axis=0)

                    pred_index = tf.math.argmax(predictions[0])
                    class_channel = tf.gather(predictions, tf.cast(pred_index, tf.int32), axis=1)
                grads = tape1.gradient(class_channel, conv_outputs)
            second_derivative = tape2.gradient(grads, conv_outputs)
            del tape2

            numerator = second_derivative
            denominator = 2.0 * second_derivative + tf.square(grads) + 1e-8
            alphas = tf.math.divide_no_nan(numerator, denominator)
            alphas = tf.nn.relu(alphas)

            weights = tf.reduce_sum(tf.maximum(grads, 0.0) * alphas, axis=(0, 1))
            conv_outputs = conv_outputs[0]
            heatmap = tf.reduce_sum(conv_outputs * weights, axis=-1)
            heatmap = tf.maximum(heatmap, 0.0)

            max_val = tf.reduce_max(heatmap)
            max_val = tf.cast(max_val, tf.float32)
            if max_val > 0:
                heatmap = heatmap / (max_val + 1e-8)

            heatmap_uint8 = np.uint8(255 * heatmap.numpy())
            heatmap_resized = cv2.resize(heatmap_uint8, (img_rgb.shape[1], img_rgb.shape[0]), interpolation=cv2.INTER_CUBIC)
            heatmap_color_bgr = cv2.applyColorMap(heatmap_resized, cv2.COLORMAP_JET)
            heatmap_color_rgb = cv2.cvtColor(heatmap_color_bgr, cv2.COLOR_BGR2RGB)
            overlay_rgb = cv2.addWeighted(img_rgb.astype(np.uint8), 0.6, heatmap_color_rgb, alpha, 0)

            return overlay_rgb.astype(np.uint8), heatmap_color_bgr.astype(np.uint8)

        except Exception as e:
            print("Grad-CAM++ failed:", e)
            return img_rgb, None

    # ---- Risk questions ----
    def ask_risk_questions(self, condition, base_conf):
        questions = self.risk_questions.get(condition, [])
        answers = {}
        if not questions:
            return base_conf, answers

        print("\nAnswer the following questions to adjust confidence:")
        for i, (q, opts) in enumerate(questions):
            print(f"\n{i+1}. {q}")
            for j, opt in enumerate(opts):
                print(f"  {j}. {opt}")
            while True:
                try:
                    choice = int(input("Enter choice number: ").strip())
                    if 0 <= choice < len(opts):
                        answers[q] = self.choice_map.get(choice, "Low")
                        break
                    else:
                        print("Invalid number. Try again.")
                except Exception:
                    print("Invalid input. Enter a number.")

        final_conf = float(base_conf)
        for ans in answers.values():
            final_conf = min(1.0, final_conf + float(self.choice_conf_map.get(ans, 0)))
        return final_conf, answers

    # ---- Run pipeline helpers ----
    def run_with_path(self, img_path):
        batch, rgb = self.preprocess_image_from_path(img_path)
        self._run_core(batch, rgb)

    def run_with_bytes(self, img_bytes):
        batch, rgb = self.preprocess_image_from_bytes(img_bytes)
        self._run_core(batch, rgb)

    def _run_core(self, batch, rgb):
        condition, base_conf = self.predict_condition(batch)
        print(f"\nPredicted Condition: {condition}")
        print(f"Base Confidence: {base_conf * 100:.2f}%")

        final_conf, answers = self.ask_risk_questions(condition, base_conf)

        if final_conf < self.conf_threshold:
            label = "Normal Eye"
            risk_note = f"AI confidence < {int(self.conf_threshold * 100)}%. Mostly normal, possible risk of {condition}."
        else:
            label = condition
            risk_note = None

        print(f"\nFinal Label: {label}")
        print(f"Final Confidence: {final_conf * 100:.2f}%")
        if answers:
            print("\nAnswers given:")
            for k, v in answers.items():
                print(f"  {k}: {v}")
        if risk_note:
            print("\nRisk Note:", risk_note)

        overlay_rgb, heatmap_bgr = self.make_gradcam_plus_plus(batch, rgb)

        # save overlay
        try:
            overlay_bgr = cv2.cvtColor(overlay_rgb, cv2.COLOR_RGB2BGR)
            cv2.imwrite("gradcam_overlay.png", overlay_bgr)
            print("\nGrad-CAM++ overlay saved as 'gradcam_overlay.png'")
        except Exception as e:
            print("Failed to save overlay:", e)

        if heatmap_bgr is not None:
            try:
                cv2.imwrite("gradcam_heatmap.png", heatmap_bgr)
                print("Grad-CAM++ heatmap saved as 'gradcam_heatmap.png'")
            except Exception as e:
                print("Failed to save heatmap:", e)


def read_stdin_bytes():
    try:
        data = sys.stdin.buffer.read()
        return data
    except Exception:
        return None

def decode_base64_maybe(s):
    s = s.strip()
    if s.startswith("data:"):
        try:
            _, payload = s.split(",", 1)
            return base64.b64decode(payload)
        except Exception:
            raise ValueError("Invalid data URL/base64 provided.")
    else:
        try:
            clean = "".join(s.split())
            return base64.b64decode(clean)
        except Exception:
            raise ValueError("Invalid base64 input.")

def prompt_model_path_loop(default="models/retinova_model.h5"):
    while True:
        mp = input(f"Enter path to model (.h5) [default: {default}]: ").strip() or default
        if os.path.exists(mp) and os.path.isfile(mp):
            return mp
        print(f"Model not found at '{mp}'. Please enter a valid model path.")

def prompt_menu_choice_loop():
    prompt = (
        "\nChoose image input method (enter number):\n"
        "  1) File path\n"
        "  2) Paste base64/data URL\n"
        "  3) Read raw image bytes from stdin (use '-' as image arg when piping)\n"
        "Enter 1, 2 or 3: "
    )
    while True:
        c = input(prompt).strip()
        if c in ("1","2","3"):
            return c
        print("Invalid choice. Please enter 1, 2 or 3.")

def prompt_image_path_loop():
    while True:
        ip = input("Enter path to image: ").strip()
        if os.path.exists(ip) and os.path.isfile(ip):
            return ip
        print(f"Image not found at '{ip}'. Please enter a valid image path.")

def prompt_base64_loop():
    while True:
        b64 = input("Paste base64 or data URL and press Enter:\n").strip()
        try:
            decoded = decode_base64_maybe(b64)
            return decoded
        except Exception as e:
            print("Invalid base64/data URL. Try again.")
def main():
    if len(sys.argv) == 1:
        model_path = prompt_model_path_loop()
        choice = prompt_menu_choice_loop()
        if choice == "1":
            image_path = prompt_image_path_loop()
            mode = "path"
        elif choice == "2":
            img_bytes = prompt_base64_loop()
            mode = "bytes"
        else:
            print("\nYou selected stdin mode. To use it, you'll need to run the script with '-' as the image argument and pipe raw bytes.")
            print("Example (Unix): cat test_eye.png | python retinova_cli.py <model.h5> -")
            while True:
                yn = input("Proceed with stdin mode now? (y/n): ").strip().lower()
                if yn in ("y","yes"):
                    image_path = "-"
                    mode = "stdin"
                    break
                elif yn in ("n","no"):
                    choice = prompt_menu_choice_loop()
                    if choice == "1":
                        image_path = prompt_image_path_loop()
                        mode = "path"
                        break
                    elif choice == "2":
                        img_bytes = prompt_base64_loop()
                        mode = "bytes"
                        break
                    else:
                        print("Stdin mode selected again; repeating prompt.")
                else:
                    print("Please answer y or n.")
    elif len(sys.argv) == 2:
        model_path = sys.argv[1]
        if not os.path.exists(model_path):
            print(f"Model not found at '{model_path}'.")
            model_path = prompt_model_path_loop()
        choice = prompt_menu_choice_loop()
        if choice == "1":
            image_path = prompt_image_path_loop()
            mode = "path"
        elif choice == "2":
            img_bytes = prompt_base64_loop()
            mode = "bytes"
        else:
            image_path = "-"
            mode = "stdin"
    else:
        model_path = sys.argv[1]
        raw_image_arg = sys.argv[2]
        if not os.path.exists(model_path):
            print(f"Model not found at '{model_path}'.")
            model_path = prompt_model_path_loop()
        if raw_image_arg == "-":
            mode = "stdin"
        elif os.path.exists(raw_image_arg):
            image_path = raw_image_arg
            mode = "path"
        else:
            try:
                img_bytes = decode_base64_maybe(raw_image_arg)
                mode = "bytes"
            except Exception:
                print(f"Image '{raw_image_arg}' not found and not valid base64.")
                choice = prompt_menu_choice_loop()
                if choice == "1":
                    image_path = prompt_image_path_loop()
                    mode = "path"
                elif choice == "2":
                    img_bytes = prompt_base64_loop()
                    mode = "bytes"
                else:
                    image_path = "-"
                    mode = "stdin"
    if not os.path.exists(model_path):
        print(f"Model not found at '{model_path}'. Exiting.")
        sys.exit(1)
    pipeline = RetiNovaCLI(model_path)
    try:
        if mode == "path":
            pipeline.run_with_path(image_path)
        elif mode == "bytes":
            pipeline.run_with_bytes(img_bytes)
        elif mode == "stdin":
            stdin_bytes = read_stdin_bytes()
            if not stdin_bytes:
                print("No data read from stdin. When piping, call (example Unix): cat image.png | python retinova_cli.py <model.h5> -")
                sys.exit(1)
            pipeline.run_with_bytes(stdin_bytes)
        else:
            print("Unknown mode. Exiting.")
            sys.exit(1)
    except Exception as e:
        print("Error during processing:", e)
        sys.exit(1)


if __name__ == "__main__":
    main()
