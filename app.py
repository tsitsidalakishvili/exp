import pathlib
from pathlib import Path

# Replace PosixPath with WindowsPath for compatibility on Windows
pathlib.PosixPath = pathlib.WindowsPath

from PIL import Image, ImageDraw, ImageEnhance, ImageOps, ImageFont
from torchvision.ops import nms
import re
import numpy as np
import cv2
import pandas as pd
import fitz
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from streamlit_tensorboard import st_tensorboard
from rapidocr_onnxruntime import RapidOCR
import io
import datetime
import streamlit as st
import os
import glob
import torch

Image.MAX_IMAGE_PIXELS = None

# Check device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Set page configuration
st.set_page_config(layout="wide")

# Set directories
base_dir = os.path.dirname(__file__)
yolov5_dir = os.path.join(base_dir, 'yolov5')
model_dir = os.path.join(yolov5_dir, 'runs', 'train')

# List available models
def list_available_models(base_dir):
    model_paths = glob.glob(os.path.join(base_dir, '**', '*.pt'), recursive=True)
    return model_paths

# Load model
@st.cache_resource
def load_model(model_path, yolov5_dir):
    model_path = str(Path(model_path).resolve())
    yolov5_dir = str(Path(yolov5_dir).resolve())
    model = torch.hub.load(yolov5_dir, 'custom', path=model_path, source='local', force_reload=True)
    model.to(device)
    return model

# Get available models
available_models = list_available_models(model_dir)
st.write("Available models:", available_models)

# Select model
selected_model = st.selectbox("Select a Model", available_models)
load_model_button = st.button("Load Model")

# Session state for model
if 'model' not in st.session_state:
    st.session_state['model'] = None

if selected_model and load_model_button:
    try:
        st.session_state['model'] = load_model(selected_model, yolov5_dir)
        st.success(f"Model '{os.path.basename(selected_model)}' loaded successfully!")
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")

# Define class color mapping
CLASS_COLORS = {
    "Instrument-square": (255, 0, 0),
    "Instrument": (0, 255, 0),
    "Instrument-offset": (0, 0, 255),
    "Instrument-square-offset": (128, 0, 128),
}

def render_pdf_page_to_png_with_mupdf(uploaded_file, dpi=300):
    images = []
    uploaded_file.seek(0)
    doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")

    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        mat = fitz.Matrix(dpi / 72, dpi / 72)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        img_bytes = pix.tobytes("png")
        img = Image.open(io.BytesIO(img_bytes))

        if img.mode != 'RGBA':
            img = img.convert('RGBA')

        images.append(img)
    return images

def run_inference_and_get_results(confidence_threshold, img, first_nms_threshold=0.3, second_nms_threshold=0.7):
    st.session_state['model'].conf = confidence_threshold
    results = st.session_state['model'](img)
    detected_objects = []
    boxes, scores = [], []

    for i in range(len(results.xyxy[0])):
        bbox = results.xyxy[0][i].cpu().numpy()
        class_id = int(bbox[5])
        class_name = st.session_state['model'].names[class_id]
        confidence = bbox[4]

        xmin, ymin, xmax, ymax = map(int, bbox[:4].tolist())
        boxes.append([xmin, ymin, xmax, ymax])
        scores.append(confidence.item())
        detected_objects.append({"class": class_name, "confidence": confidence.item(), "bbox": [xmin, ymin, xmax, ymax]})

    if not boxes:
        return []

    boxes_tensor = torch.tensor(boxes, dtype=torch.float32)
    scores_tensor = torch.tensor(scores, dtype=torch.float32)

    keep_indices_first = nms(boxes_tensor, scores_tensor, first_nms_threshold)
    if not len(keep_indices_first):
        return []

    final_indices = keep_indices_first if second_nms_threshold >= first_nms_threshold else keep_indices_first[nms(boxes_tensor[keep_indices_first], scores_tensor[keep_indices_first], second_nms_threshold)]
    detected_objects_nms = [detected_objects[i] for i in final_indices]
    return detected_objects_nms

def draw_boxes_with_class_colors(image, detections):
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    for detection in detections:
        class_name = detection['class']
        confidence = detection['confidence']
        xmin, ymin, xmax, ymax = detection['bbox']
        color = CLASS_COLORS.get(class_name, (255, 255, 255))
        draw.rectangle([xmin, ymin, xmax, ymax], outline=color, width=2)
        text = f"{class_name} {confidence:.2f}"
        draw.text((xmin, ymin), text, fill=color, font=font)
    return image

def crop_detected_areas(image, detections, margin=5):
    cropped_images = []
    for det in detections:
        xmin, ymin, xmax, ymax = det['bbox']
        xmin, ymin = max(0, xmin - margin), max(0, ymin - margin)
        xmax, ymax = min(image.width, xmax + margin), min(image.height, ymax + margin)
        if xmin < xmax and ymin < ymax:
            cropped_images.append(image.crop((xmin, ymin, xmax, ymax)))
    return cropped_images

def enhance_images(images, resize_factor, denoise_strength, denoise_template_window_size, denoise_search_window, thresholding, deskew_angle):
    enhanced_images = []
    for image in images:
        new_size = (int(image.width * resize_factor), int(image.height * resize_factor))
        resized_image = image.resize(new_size, Image.LANCZOS)
        grayscale_image = ImageOps.grayscale(resized_image)
        np_grayscale = np.array(grayscale_image)
        denoised_image = cv2.fastNlMeansDenoising(np_grayscale, None, denoise_strength, denoise_template_window_size, denoise_search_window)
        binarized_image = cv2.threshold(denoised_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1] if thresholding else denoised_image

        coords = np.column_stack(np.where(binarized_image > 0))
        angle = cv2.minAreaRect(coords)[-1]
        angle = -(90 + angle) if angle < -45 else (90 - angle) if angle > 45 else -angle
        angle += deskew_angle
        M = cv2.getRotationMatrix2D((binarized_image.shape[1] // 2, binarized_image.shape[0] // 2), angle, 1.0)
        deskewed_image = cv2.warpAffine(binarized_image, M, (binarized_image.shape[1], binarized_image.shape[0]), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

        sharpened_image = cv2.filter2D(deskewed_image, -1, np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]]))
        enhanced_images.append(Image.fromarray(sharpened_image))
    return enhanced_images

class RapidOCRTextExtractor:
    def __init__(self, engine):
        self.engine = engine

    def extract_text(self, image):
        open_cv_image = pil_to_cv2(image)
        preprocessed_image = preprocess_for_ocr(open_cv_image)
        result, _ = self.engine(preprocessed_image)
        return ' '.join([res[1] for res in result]) if result else ''

def pil_to_cv2(pil_image):
    return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

def preprocess_for_ocr(image, target_size=(300, 300)):
    image = cv2.resize(image, target_size)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.adaptiveThreshold(gray_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

ocr_engine = RapidOCR()
text_extractor = RapidOCRTextExtractor(ocr_engine)

def extract_text_from_pdf(uploaded_file):
    uploaded_file.seek(0)
    doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
    text = ""
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        text += page.get_text("text") + "\n"
    return text

if 'extracted_data' not in st.session_state:
    st.session_state['extracted_data'] = []

def generate_regex_pattern(parts):
    pattern_map = {
        "System Number": r"(\d{2})",
        "Function Code": r"([A-Z]{2,5})",
        "Loop Sequence": r"(\d{4})"
    }
    selected_patterns = [pattern_map[part] for part in parts]
    return r"\b" + r"\s*".join(selected_patterns) + r"\b"

page = st.sidebar.selectbox("Select a page", ("About Project", "Object Detection", "OCR & RegEx"))

# About Project Page
if page == "About Project":
    st.write("## About Project 📖")
    st.write("""
    ### Project Overview
    This tool is designed to assist with the detection and extraction of instrument and equipment tagnames from Piping and Instrumentation Diagrams (P&IDs).
    The primary goal is to automate the process of identifying and extracting tagnames, reducing manual effort, and increasing accuracy.

    ### The Goal and Requirements
    For a folder full of P&IDs in PDF format, the tool creates a set of CSV files with a list of all the equipment and instruments on each drawing.
    These CSVs will be used by our Knowledge Graph build scripts. The output CSV file contains the following columns:
    - Tagname
    - Class (the type detected by the AI)
    - System
    - Function_Code
    - Drawing_No

    ### How the Tool Works
    1. Upload a document or point it to a folder with lots of documents.
    2. Provide configuration input to tell the algorithm the structure of Equipment and Instrumentation tags.
    3. For each document, the tool makes two passes:
        - Straight OCR to find all the text, then use Regex to extract valid Equipment and Instrument tagnames.
        - Use image recognition to find the shapes of the instruments and then extract the text from these.
    4. Combine the output from the two passes, remove duplicates, and output the final CSV file.
    """)

if page == "Object Detection":
    st.write("## Object Detection 🔍\nUse the Object Detection feature to automatically identify and label different instruments and components in your P&ID diagrams. Adjust detection settings as needed.")
    tab_option = st.radio("Select Option", ["Detect Object", "Image Enhancement Tool"], horizontal=True)

    if tab_option == "TensorBoard":
        st.write("## TensorBoard Integration 📊\nView TensorBoard logs and visualize the training metrics and other relevant data.")
        st_tensorboard(logdir=logdir, port=6006, width=1080)

    if tab_option == "Model Performance":
        st.write("## Model Performance Metrics")
        metrics = extract_metrics(logdir)
        display_metrics(metrics)

    if tab_option == "Detect Object":
        dpi = st.number_input("Select DPI for PDF Rendering", min_value=100, max_value=600, value=300, step=50)
        uploaded_files = st.file_uploader("Choose images or PDFs...", type=["jpg", "png", "jpeg", "pdf"], accept_multiple_files=True)
        confidence_threshold = st.slider("Select Confidence Threshold", min_value=0.0, max_value=1.0, value=0.05, step=0.01)
        images = []
        extracted_texts = []

        if uploaded_files:
            for uploaded_file in uploaded_files:
                if uploaded_file.type == "application/pdf":
                    with st.spinner(f"Processing {uploaded_file.name}..."):
                        images.extend(render_pdf_page_to_png_with_mupdf(uploaded_file, dpi=dpi))
                        uploaded_file.seek(0)
                        extracted_texts.append(extract_text_from_pdf(uploaded_file))
                else:
                    image = Image.open(uploaded_file).convert('RGB')
                    images.append(image)
                    extracted_texts.append(text_extractor.extract_text(image))

            st.session_state['images'] = images
            st.session_state['extracted_texts'] = extracted_texts

            detect_objects = st.button('Detect Objects')
            if detect_objects:
                if st.session_state['model'] is not None:
                    total_detections = 0
                    for image in images:
                        detections = run_inference_and_get_results(confidence_threshold, image)
                        if detections:
                            image_with_boxes = draw_boxes_with_class_colors(image.copy(), detections)
                            st.image(image_with_boxes, caption='Detected Objects', use_column_width=True)
                            st.session_state['detected_objects'] = detections
                            st.session_state['images_with_boxes'] = image_with_boxes
                            total_detections += len(detections)
                        else:
                            st.warning("No objects detected.")
                    st.write(f"{total_detections} Objects Detected")
                else:
                    st.warning("Please load a model before detecting objects.")

        if 'detected_objects' in st.session_state and st.session_state['detected_objects']:
            margin = st.slider("Select Margin Size for Cropping (pixels)", min_value=0, max_value=50, value=5, step=1, key="crop_margin_slider")
            if st.button('Save Detections'):
                cropped_images_list, cropped_images_paths = [], []
                cropped_images_dir = "cropped_images"
                os.makedirs(cropped_images_dir, exist_ok=True)

                for idx, image in enumerate(images):
                    cropped_images = crop_detected_areas(image, st.session_state['detected_objects'], margin=margin)
                    for crop_idx, cropped_image in enumerate(cropped_images):
                        cropped_image_path = os.path.join(cropped_images_dir, f"cropped_image_{idx}_{crop_idx}.png")
                        cropped_image.save(cropped_image_path)
                        cropped_images_list.append(cropped_image)
                        cropped_images_paths.append(cropped_image_path)

                st.session_state['cropped_images_paths'] = cropped_images_paths
                st.write(f"{len(cropped_images_list)} detected objects cropped and saved.")
                if cropped_images_list:
                    num_to_display = min(len(cropped_images_list), 7)
                    cropped_cols = st.columns(num_to_display)
                    for idx, cropped_col in enumerate(cropped_cols):
                        if idx < len(cropped_images_list):
                            cropped_col.image(cropped_images_list[idx], caption=f'Cropped Object {idx + 1}', width=100)

    if tab_option == "Image Enhancement Tool":
        with st.expander("Improve Detected Objects Quality"):
            if 'cropped_images_paths' in st.session_state and len(st.session_state['cropped_images_paths']) > 0:
                st.success("Detected objects:")
                cols = st.columns(len(st.session_state['cropped_images_paths']))
                for idx, col in enumerate(cols):
                    with col:
                        image = Image.open(st.session_state['cropped_images_paths'][idx])
                        st.image(image, caption=f'Detected Object {idx + 1}', width=100)

                col1, col2 = st.columns(2)
                with col1:
                    resize_factor = st.slider("Resize Factor", min_value=0.1, max_value=9.0, value=1.0, step=0.1, key='resize_factor1')
                    denoise_strength = st.slider("Denoise Strength", min_value=0, max_value=200, value=10, key='denoise_strength1')
                    denoise_template_window_size = st.slider("Denoise Template Window Size", min_value=3, max_value=41, value=7, step=2, key='denoise_template_window_size1')
                    denoise_search_window = st.slider("Denoise Search Window", min_value=3, max_value=41, value=21, step=2, key='denoise_search_window1')
                    thresholding = st.checkbox("Thresholding", value=True, key='thresholding1')
                    deskew_angle = st.slider("Deskew Angle", min_value=-90, max_value=90, value=0, key='deskew_angle1')

                    if st.button('Apply Enhancements', key='apply_enhancements1'):
                        enhanced_images = enhance_images(
                            [Image.open(path) for path in st.session_state['cropped_images_paths']],
                            resize_factor=resize_factor,
                            denoise_strength=denoise_strength,
                            denoise_template_window_size=denoise_template_window_size,
                            denoise_search_window=denoise_search_window,
                            thresholding=thresholding,
                            deskew_angle=deskew_angle
                        )
                        st.session_state['enhanced_images'] = enhanced_images
                        st.success("Enhancement parameters applied successfully.")

                with col2:
                    if 'enhanced_images' in st.session_state and len(st.session_state['enhanced_images']) > 0:
                        st.image(st.session_state['enhanced_images'][0], caption='Enhanced Object 1')
            else:
                st.warning("No cropped images to display. Please detect objects first.")

if page == "OCR & RegEx":
    st.write("## OCR (Optical Character Recognition) & RegEx 📖\nUse OCR to extract text from detected objects in your images. Customize your extraction with naming conventions and regular expressions for precise analysis.")

    with st.expander("Extract Instruments"):
        st.subheader("Naming Convention Builder")
        st.write("Build your naming convention by selecting the order of components and separators.")

        parts = ["None", "System Number", "Function Code", "Loop Sequence"]
        separators = ["-", ""]

        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            part1 = st.selectbox("First Part", parts, index=2, key="part1")
            if part1 == "System Number":
                system_hint1 = st.text_input("Enter System Number Hint:", value='', key="system_hint1")
        with col2:
            sep1 = st.selectbox("Separator After First Part", separators, key="sep1")
        with col3:
            part2 = st.selectbox("Second Part", parts, index=2, key="part2")
            if part2 == "System Number":
                system_hint2 = st.text_input("Enter System Number Hint:", value='13', key="system_hint2")
        with col4:
            sep2 = st.selectbox("Separator After Second Part", separators, key="sep2")
        with col5:
            part3 = st.selectbox("Third Part", parts, index=2, key="part3")
            if part3 == "System Number":
                system_hint3 = st.text_input("Enter System Number Hint:", value='13', key="system_hint3")

        selected_parts = [part1, part2, part3]
        selected_separators = [sep1, sep2]

        system_number_pattern = r'\b(\d{2})\b'
        function_code_pattern = r'\b([A-Z]{2,5})\b'
        loop_sequence_pattern = r'\b(\d{4})\b'

        if st.button('Extract Instruments'):
            extracted_data = []

            if 'cropped_images_paths' in st.session_state:
                for image_path in st.session_state['cropped_images_paths']:
                    text = text_extractor.extract_text(Image.open(image_path))

                    system_number_matches = re.findall(system_number_pattern, text)
                    function_code_matches = re.findall(function_code_pattern, text)
                    loop_sequence_matches = re.findall(loop_sequence_pattern, text)

                    file_name_match = re.search(r'(.+?)\.png', os.path.basename(image_path))
                    drawing_no = file_name_match.group(1) if file_name_match else "Unknown"

                    for function_code, loop_sequence in zip(function_code_matches, loop_sequence_matches):
                        system_number = system_number_matches[0] if system_number_matches else (
                            system_hint1 if part1 == "System Number" else (
                                system_hint2 if part2 == "System Number" else (
                                    system_hint3 if part3 == "System Number" else ""
                                )
                            )
                        )

                        parts_dict = {
                            "System Number": system_number,
                            "Function Code": function_code,
                            "Loop Sequence": loop_sequence
                        }

                        tagname_parts = []
                        for i, part in enumerate(selected_parts):
                            if part != "None":
                                tagname_parts.append(parts_dict[part])
                                if i < len(selected_separators):
                                    tagname_parts.append(selected_separators[i])
                        tagname = ''.join(tagname_parts).rstrip("-")

                        extracted_data.append({
                            'Tagname': tagname,
                            'Class': 'INSTRUMENT',
                            'System': system_number,
                            'Function_Code': function_code,
                            'Loop_Sequence': loop_sequence,
                            'Drawing_No': drawing_no
                        })

                st.divider()
                df = pd.DataFrame(extracted_data)
                st.divider()
                with st.container():
                    col1, col2 = st.columns(2)
                    with col2:
                        st.write("### Uploaded P&ID Diagram")
                        if 'images_with_boxes' in st.session_state and st.session_state['images_with_boxes']:
                            st.image(st.session_state['images_with_boxes'], caption='Uploaded Image with Detections')
                    with col1:
                        if not df.empty:
                            st.write("### Extracted Instruments Data")
                            st.dataframe(df)
                            csv_path = 'output_instruments.csv'
                            df.to_csv(csv_path, sep=';', index=False)
                            st.download_button(
                                label="Download CSV",
                                data=df.to_csv(index=False).encode('utf-8'),
                                file_name='output_instruments.csv',
                                mime='text/csv'
                            )
                        else:
                            st.warning("No matches found or no images to process.")
            else:
                st.warning("No detected objects or cropped images found in the session state.")

    with st.expander("Extract Other Equipments"):
        if st.button('Extract'):
            extracted_equipments = []

            if 'extracted_texts' in st.session_state:
                for page_num, text in enumerate(st.session_state['extracted_texts']):
                    vessel_pattern = r'\bV[A-Z]-\d{2}-\d{3}\b'
                    supply_lines_pattern = r'\b\d{4}-[A-Z]{3}-\d{2}-\d{4}-[A-Z]{2}\d?-\d{2}-[A-Z]\b'
                    vessel_matches = re.findall(vessel_pattern, text)
                    supply_lines_matches = re.findall(supply_lines_pattern, text)
                    for vessel_match in vessel_matches:
                        tagname = vessel_match
                        extracted_equipments.append({
                            'Tagname': tagname,
                            'Class': 'VESSEL',
                            'Drawing_No': f'Page {page_num + 1}'
                        })
                    for supply_line in supply_lines_matches:
                        extracted_equipments.append({
                            'Tagname': supply_line,
                            'Class': 'SUPPLY LINE',
                            'Drawing_No': f'Page {page_num + 1}'
                        })

                df_equip = pd.DataFrame(extracted_equipments)
                st.divider()
                with st.container():
                    if not df_equip.empty:
                        st.write("### Extracted Equipment Data")
                        st.dataframe(df_equip)
                        csv_path_equip = 'output_equipments.csv'
                        df_equip.to_csv(csv_path_equip, sep=';', index=False)
                        st.download_button(
                            label="Download CSV",
                            data=df_equip.to_csv(index=False).encode('utf-8'),
                            file_name='output_equipments.csv',
                            mime='text/csv'
                        )
                    else:
                        st.warning("No matches found.")
            else:
                st.warning("No text extracted yet. Please upload a file on the Object Detection page.")

    with st.expander("Complete Extracted Text"):
        if 'extracted_texts' in st.session_state:
            extracted_texts = st.session_state['extracted_texts']
            for i, text in enumerate(extracted_texts):
                st.write(f"**Page {i + 1}:**")
                st.write(text)
        else:
            st.warning("No text extracted yet. Please upload a file on the Object Detection page.")

    with st.expander("Combined Extracted Instruments and Equipments"):
        if 'extracted_data' in st.session_state or 'extracted_texts' in st.session_state:
            combined_data = []

            if 'extracted_data' in st.session_state:
                combined_data.extend(st.session_state['extracted_data'])

            if 'extracted_texts' in st.session_state:
                for page_num, text in enumerate(st.session_state['extracted_texts']):
                    vessel_pattern = r'\bV[A-Z]-\d{2}-\d{3}\b'
                    supply_lines_pattern = r'\b\d{4}-[A-Z]{3}-\d{2}-\d{4}-[A-Z]{2}\d?-\d{2}-[A-Z]\b'
                    vessel_matches = re.findall(vessel_pattern, text)
                    supply_lines_matches = re.findall(supply_lines_pattern, text)
                    for vessel_match in vessel_matches:
                        tagname = vessel_match
                        combined_data.append({
                            'Tagname': tagname,
                            'Class': 'VESSEL',
                            'Drawing_No': f'Page {page_num + 1}'
                        })
                    for supply_line in supply_lines_matches:
                        combined_data.append({
                            'Tagname': supply_line,
                            'Class': 'SUPPLY LINE',
                            'Drawing_No': f'Page {page_num + 1}'
                        })

            df_combined = pd.DataFrame(combined_data)
            if not df_combined.empty:
                st.write("### Combined Extracted Data")
                st.dataframe(df_combined)
                csv_path_combined = 'output_combined.csv'
                df_combined.to_csv(csv_path_combined, sep=';', index=False)
                st.download_button(
                    label="Download Combined CSV",
                    data=df_combined.to_csv(index=False).encode('utf-8'),
                    file_name='output_combined.csv',
                    mime='text/csv'
                )
            else:
                st.warning("No combined data found.")
        else:
            st.warning("No extracted data found. Please process some files first.")
