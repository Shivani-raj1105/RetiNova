# retinova_cli.py
"""
Robust CLI wrapper for the RetiNova model.
- Loads Keras .h5 model
- Preprocesses image (resizes to 224x224 and scales 0..1)
- Predicts class + base confidence
- Asks risk questions for the predicted condition (adjusts confidence)
- Produces Grad-CAM++ heatmap and overlay and saves:
    - gradcam_heatmap.png  (colored heatmap, BGR saved by OpenCV)
    - gradcam_overlay.png  (original image blended with heatmap)
This file is defensive: handles many edge cases of model outputs and input shapes.
"""

import os
import sys
import warnings
import logging
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

        # class labels used by your model
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

        # question/confidence maps
        self.choice_map = {0: "Low", 1: "Mild", 2: "Moderate", 3: "High"}
        self.choice_conf_map = {"Low": 0.0, "Mild": 0.05, "Moderate": 0.1, "High": 0.2}

        # load model defensively (avoid compile side effects)
        try:
            self.model = load_model(model_path, compile=False)
        except Exception as e:
            raise RuntimeError(f"Failed to load model from '{model_path}': {e}")

        # find the last conv-like layer robustly
        self.last_conv_layer_name = None
        try:
            for layer in reversed(self.model.layers):
                # robust test for convolutional layers
                lname = layer.__class__.__name__.lower()
                if "conv" in lname or "separableconv" in lname or "depthwiseconv" in lname:
                    self.last_conv_layer_name = layer.name
                    break
            # fallback: search by name containing 'conv'
            if self.last_conv_layer_name is None:
                for layer in reversed(self.model.layers):
                    if 'conv' in layer.name.lower():
                        self.last_conv_layer_name = layer.name
                        break
        except Exception:
            self.last_conv_layer_name = None

        # full risk question bank (as you provided)
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

    # ---------------- Image preprocessing ----------------
    def preprocess_image(self, img_path):
        """
        Read image via OpenCV, convert to RGB, resize and normalize -> return (1,H,W,3) float32 and original RGB uint8.
        """
        bgr = cv2.imread(img_path)
        if bgr is None:
            raise ValueError(f"Could not read image at {img_path}")
        # keep original RGB for overlay visualization
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, self.img_size, interpolation=cv2.INTER_AREA)
        norm = resized.astype(np.float32) / 255.0
        # ensure shape (1,H,W,3)
        batch = np.expand_dims(norm, axis=0)
        return batch, rgb.astype(np.uint8)

    # ---------------- Prediction ----------------
    def predict_condition(self, img_array):
        """
        Run model.predict and handle varied return shapes (numpy arrays, lists, tuples).
        Returns: (condition_label, base_confidence_float)
        """
        # ensure numpy array
        img_array = np.array(img_array, dtype=np.float32)
        if img_array.ndim == 3:
            img_array = np.expand_dims(img_array, axis=0)

        preds = self.model.predict(img_array, verbose=0)

        # If model returns list/tuple (multiple outputs), pick the last element (usually logits/probs)
        if isinstance(preds, (list, tuple)):
            preds = preds[-1]

        preds = np.array(preds)

        # If predictions is 1D (single sample vector), make it 2D
        if preds.ndim == 1:
            preds = np.expand_dims(preds, axis=0)

        # safety: if model gave scalar (single-value), treat as single-class probability
        if preds.size == 1:
            idx = 0
            conf = float(preds.flatten()[0])
        else:
            # take argmax across last axis for first sample
            idx = int(np.argmax(preds[0]))
            conf = float(preds[0, idx])

        condition = self.conditions[idx] if idx < len(self.conditions) else "Unknown"
        return condition, conf

    # ---------------- Grad-CAM++ ----------------
    def make_gradcam_plus_plus(self, img_array, img_rgb, alpha=0.4):
        """
        Return (overlay_rgb_uint8, heatmap_bgr_uint8 or None)
        heatmap_bgr_uint8 is suitable for cv2.imwrite directly.
        """

        # Defensive conversions to numpy/tensor shapes
        try:
            img_np = np.array(img_array, dtype=np.float32)
            if img_np.ndim == 3:
                img_np = np.expand_dims(img_np, axis=0)
        except Exception:
            # fallback
            print("Grad-CAM++: could not coerce input to numpy array; skipping heatmap.")
            return img_rgb, None

        if self.last_conv_layer_name is None:
            # no conv layer discovered
            return img_rgb, None

        try:
            # Build a small model that outputs conv features and model predictions
            conv_layer = self.model.get_layer(self.last_conv_layer_name)
            grad_model = tf.keras.models.Model(inputs=self.model.input,
                                              outputs=[conv_layer.output, self.model.output])

            img_tensor = tf.convert_to_tensor(img_np, dtype=tf.float32)

            # Use gradient tapes to compute first and second derivatives
            with tf.GradientTape(persistent=True) as tape2:
                with tf.GradientTape() as tape1:
                    conv_outputs, predictions = grad_model(img_tensor, training=False)

                    # If predictions is a list/tuple tensor-like, pick last element
                    if isinstance(predictions, (list, tuple)):
                        predictions = predictions[-1]

                    # Ensure predictions is a tensor with shape (batch, classes)
                    predictions = tf.convert_to_tensor(predictions)
                    # If predictions is rank 0 or 1, reshape safely to (batch, classes)
                    if predictions.shape.rank == 0:
                        predictions = tf.reshape(predictions, (1, 1))
                    elif predictions.shape.rank == 1:
                        # shape (classes,) -> (1, classes)
                        predictions = tf.expand_dims(predictions, axis=0)

                    # choose predicted index for the first sample
                    pred_index = tf.math.argmax(predictions[0])
                    # gather class channel safely -> shape (batch, 1)
                    class_channel = tf.gather(predictions, tf.cast(pred_index, tf.int32), axis=1)
                    # class_channel has shape (batch,1), keep as is for gradients
                # gradients of the class score w.r.t conv feature maps
                grads = tape1.gradient(class_channel, conv_outputs)
            second_derivative = tape2.gradient(grads, conv_outputs)
            del tape2

            # compute alpha coefficients per Grad-CAM++ formula (vectorized)
            numerator = second_derivative
            denominator = 2.0 * second_derivative + tf.square(grads) + 1e-8
            alphas = tf.math.divide_no_nan(numerator, denominator)
            alphas = tf.nn.relu(alphas)

            # weights: sum over spatial dims
            weights = tf.reduce_sum(tf.maximum(grads, 0.0) * alphas, axis=(0, 1))
            conv_outputs = conv_outputs[0]  # remove batch dim -> (H, W, C)
            heatmap = tf.reduce_sum(conv_outputs * weights, axis=-1)  # (H, W)
            heatmap = tf.maximum(heatmap, 0.0)

            max_val = tf.reduce_max(heatmap)
            max_val = tf.cast(max_val, tf.float32)
            if max_val > 0:
                heatmap = heatmap / (max_val + 1e-8)

            # convert to uint8 grayscale
            heatmap_uint8 = np.uint8(255 * heatmap.numpy())

            # resize to original image size (img_rgb is original RGB uint8)
            heatmap_resized = cv2.resize(heatmap_uint8, (img_rgb.shape[1], img_rgb.shape[0]), interpolation=cv2.INTER_CUBIC)

            # colorize (OpenCV returns BGR)
            heatmap_color_bgr = cv2.applyColorMap(heatmap_resized, cv2.COLORMAP_JET)

            # convert heatmap from BGR -> RGB for overlay to match img_rgb (which is RGB)
            heatmap_color_rgb = cv2.cvtColor(heatmap_color_bgr, cv2.COLOR_BGR2RGB)

            # overlay (blend)
            overlay_rgb = cv2.addWeighted(img_rgb.astype(np.uint8), 0.6, heatmap_color_rgb, alpha, 0)

            # return overlay (RGB uint8) and heatmap (BGR uint8 for saving directly)
            return overlay_rgb.astype(np.uint8), heatmap_color_bgr.astype(np.uint8)

        except Exception as e:
            # defensive fallback: print short error and return original image
            print("Grad-CAM++ failed:", e)
            return img_rgb, None

    # ---------------- Risk questions ----------------
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

    # ---------------- Run pipeline ----------------
    def run(self, img_path):
        # Preprocess
        img_array, img_rgb = self.preprocess_image(img_path)

        # Predict
        condition, base_conf = self.predict_condition(img_array)
        print(f"\nPredicted Condition: {condition}")
        print(f"Base Confidence: {base_conf * 100:.2f}%")

        # Risk questions
        final_conf, answers = self.ask_risk_questions(condition, base_conf)

        # Final label
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

        # Grad-CAM++ -> overlay (RGB) and heatmap (BGR)
        overlay_rgb, heatmap_bgr = self.make_gradcam_plus_plus(img_array, img_rgb)

        # Save overlay (convert RGB -> BGR for OpenCV)
        try:
            overlay_bgr = cv2.cvtColor(overlay_rgb, cv2.COLOR_RGB2BGR)
            cv2.imwrite("gradcam_overlay.png", overlay_bgr)
            print("\nGrad-CAM++ overlay saved as 'gradcam_overlay.png'")
        except Exception as e:
            print("Failed to save overlay:", e)

        # Save heatmap if produced
        if heatmap_bgr is not None:
            try:
                cv2.imwrite("gradcam_heatmap.png", heatmap_bgr)
                print("Grad-CAM++ heatmap saved as 'gradcam_heatmap.png'")
            except Exception as e:
                print("Failed to save heatmap:", e)


# ---------------- CLI entrypoint ----------------
def main():
    # allow interactive mode if no args
    if len(sys.argv) == 1:
        model_path = input("Enter path to model (.h5) [default: models/retinova_model.h5]: ").strip() or "models/retinova_model.h5"
        image_path = input("Enter path to image: ").strip()
    elif len(sys.argv) == 2:
        model_path = sys.argv[1]
        image_path = input("Enter path to image: ").strip()
    else:
        model_path = sys.argv[1]
        image_path = sys.argv[2]

    if not os.path.exists(model_path):
        print(f"Model not found at '{model_path}'")
        sys.exit(1)
    if not os.path.exists(image_path):
        print(f"Image not found at '{image_path}'")
        sys.exit(1)

    pipeline = RetiNovaCLI(model_path)
    pipeline.run(image_path)


if __name__ == "__main__":
    main()
