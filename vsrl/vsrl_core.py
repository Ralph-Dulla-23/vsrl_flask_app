import os
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import cv2
import random
from datetime import datetime
import time
import requests

# For Tesseract OCR
import pytesseract
# Set Tesseract path for Windows
if os.name == 'nt':  # Windows
    tesseract_path = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
    if os.path.exists(tesseract_path):
        pytesseract.pytesseract.tesseract_cmd = tesseract_path

# Create output directory
OUTPUT_DIR = "static/results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Global variable to store loaded VSRL data
VSRL_DATA = None

def load_vsrl_data(data_dir="C:\\GithubRepos\\TechWrite\\vsrl_flask_app\\vsrl\\data\\VSRL"):
    """Load all VSRL JSON files and index them by subject and content"""
    vsrl_data = {
        "by_id": {},
        "by_subject": {
            "biology": [],
            "physics": [],
            "chemistry": [],
            "general": []
        },
        "by_keyword": {}
    }
    
    # Load all JSON files
    try:
        for filename in os.listdir(data_dir):
            if filename.endswith('.json'):
                file_path = os.path.join(data_dir, filename)
                with open(file_path, 'r') as f:
                    data = json.load(f)
                
                # Get image ID from filename (remove extension)
                image_id = os.path.splitext(filename)[0]
                if image_id.startswith('vsrl_'):
                    image_id = image_id[5:]  # Remove 'vsrl_' prefix if present
                
                # Store by ID
                vsrl_data["by_id"][image_id] = data
                
                # Store by subject
                subject = data.get("educational_context", {}).get("subject", "general")
                if subject in vsrl_data["by_subject"]:
                    vsrl_data["by_subject"][subject].append(image_id)
                
                # Index by keywords (from text entities)
                text_content = data.get("educational_context", {}).get("text_content", "")
                if text_content:
                    words = [w.lower() for w in text_content.split()]
                    for word in words:
                        if len(word) > 3:  # Only index meaningful words
                            if word not in vsrl_data["by_keyword"]:
                                vsrl_data["by_keyword"][word] = []
                            if image_id not in vsrl_data["by_keyword"][word]:
                                vsrl_data["by_keyword"][word].append(image_id)
        
        print(f"Loaded {len(vsrl_data['by_id'])} VSRL annotations")
        return vsrl_data
    
    except Exception as e:
        print(f"Error loading VSRL data: {e}")
        return {"by_id": {}, "by_subject": {}, "by_keyword": {}}

def initialize_vsrl_data():
    global VSRL_DATA
    # Adjust the path to where your VSRL JSON files are stored
    VSRL_DATA = load_vsrl_data("C:\\GithubRepos\\TechWrite\\vsrl_flask_app\\vsrl\\data\\VSRL")

# Initialize VSRL data at module import
initialize_vsrl_data()

def get_vsrl_visualization_path(image_id):
    """Get the path to a pre-existing VSRL visualization"""
    visualization_path = f"C:\\GithubRepos\\TechWrite\\vsrl_flask_app\\vsrl\\data\\VSRL\\visualizations\\{image_id}.png"
    if os.path.exists(visualization_path):
        return visualization_path
    return None

# Optimized visual element detection
def detect_visual_elements(image_path):
    """Detect visual elements with optimized performance"""
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    height, width = img.shape[:2]

    detected_elements = {
        'text': [],
        'objects': [],
        'connectors': []
    }

    # Text detection using OCR with optimized parameters
    try:
        import pytesseract
        # Use a custom configuration to speed up OCR
        custom_config = r'--oem 3 --psm 11'
        text_results = pytesseract.image_to_data(img_rgb, output_type=pytesseract.Output.DICT, config=custom_config)

        for i in range(len(text_results['text'])):
            if int(text_results['conf'][i]) > 50 and text_results['text'][i].strip():
                x = text_results['left'][i]
                y = text_results['top'][i]
                w = text_results['width'][i]
                h = text_results['height'][i]
                text = text_results['text'][i]

                detected_elements['text'].append({
                    'id': f"T{len(detected_elements['text'])}",
                    'type': 'text',
                    'bbox': [[x, y], [x+w, y+h]],
                    'value': text,
                    'confidence': int(text_results['conf'][i]) / 100
                })
    except Exception as e:
        print(f"OCR detection error: {e}")

    # Object detection using optimized contour approach
    try:
        # Convert to grayscale for faster processing
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Use adaptive thresholding for better results
        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                      cv2.THRESH_BINARY_INV, 11, 2)

        # Find contours with optimized parameters
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        min_area = (width * height) * 0.005
        max_objects = 25  # Limit to top 25 objects

        # Sort contours by area and take the largest ones
        sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)[:max_objects]

        for i, contour in enumerate(sorted_contours):
            area = cv2.contourArea(contour)
            if area > min_area:
                # Check if this contour overlaps with text
                x, y, w, h = cv2.boundingRect(contour)
                is_text_container = False

                for text_elem in detected_elements['text']:
                    tx1, ty1 = text_elem['bbox'][0]
                    tx2, ty2 = text_elem['bbox'][1]
                    # If text is mostly inside this contour, it's likely a text container
                    if tx1 > x and ty1 > y and tx2 < x+w and ty2 < y+h:
                        is_text_container = True
                        break

                if not is_text_container:
                    # Simplify the polygon for efficiency
                    epsilon = 0.005 * cv2.arcLength(contour, True)
                    approx = cv2.approxPolyDP(contour, epsilon, True)

                    polygon = []
                    for point in approx:
                        polygon.append([int(point[0][0]), int(point[0][1])])

                    detected_elements['objects'].append({
                        'id': f"B{len(detected_elements['objects'])}",
                        'type': 'blob',
                        'polygon': polygon,
                        'area': area
                    })
    except Exception as e:
        print(f"Object detection error: {e}")

    # Enhanced connector detection
    try:
        # Use bilateral filter to reduce noise while preserving edges
        filtered = cv2.bilateralFilter(gray, 9, 75, 75)

        # Enhanced edge detection
        edges = cv2.Canny(filtered, 50, 150, apertureSize=3)

        # Dilate edges to connect broken lines
        kernel = np.ones((3,3), np.uint8)
        dilated_edges = cv2.dilate(edges, kernel, iterations=1)

        # Improved line detection with better parameters
        lines = cv2.HoughLinesP(dilated_edges, 1, np.pi/180, threshold=80,
                               minLineLength=30, maxLineGap=15)

        if lines is not None:
            # Filter lines to find potential connectors/arrows
            potential_connectors = []

            for i, line in enumerate(lines):
                x1, y1, x2, y2 = line[0]

                length = np.sqrt((x2-x1)**2 + (y2-y1)**2)
                angle = np.arctan2(y2-y1, x2-x1) * 180 / np.pi

                # Filter diagonal lines (more likely to be connectors)
                if 15 < abs(angle) < 75 or 105 < abs(angle) < 165:
                    potential_connectors.append({
                        'line': line[0],
                        'length': length,
                        'angle': angle
                    })

            # Sort by length and take the top candidates
            potential_connectors.sort(key=lambda x: x['length'], reverse=True)
            max_connectors = min(15, len(potential_connectors))  # Limit to 15 connectors

            for i, connector in enumerate(potential_connectors[:max_connectors]):
                x1, y1, x2, y2 = connector['line']
                angle = connector['angle']

                # Create a polygon representation
                line_width = 5
                dx = line_width * np.sin(angle * np.pi / 180)
                dy = line_width * np.cos(angle * np.pi / 180)

                polygon = [
                    [int(x1 + dx), int(y1 - dy)],
                    [int(x1 - dx), int(y1 + dy)],
                    [int(x2 - dx), int(y2 + dy)],
                    [int(x2 + dx), int(y2 - dy)]
                ]

                # Check for arrowhead at the end point
                has_arrowhead = False
                arrowhead_region = img[max(0, y2-20):min(height, y2+20), max(0, x2-20):min(width, x2+20)]
                if arrowhead_region.size > 0:
                    # Simple heuristic: check for darker pixels that might form arrowhead
                    gray_region = cv2.cvtColor(arrowhead_region, cv2.COLOR_BGR2GRAY) if len(arrowhead_region.shape) > 2 else arrowhead_region
                    has_arrowhead = np.mean(gray_region) < 200  # Arbitrary threshold

                detected_elements['connectors'].append({
                    'id': f"A{len(detected_elements['connectors'])}",
                    'type': 'arrow' if has_arrowhead else 'connector',
                    'polygon': polygon,
                    'start': [x1, y1],
                    'end': [x2, y2],
                    'has_arrowhead': has_arrowhead
                })
    except Exception as e:
        print(f"Connector detection error: {e}")

    print(f"Detected {len(detected_elements['text'])} text elements, " +
          f"{len(detected_elements['objects'])} objects, " +
          f"{len(detected_elements['connectors'])} connectors")

    return detected_elements

# Infer relationships
def infer_relationships(elements):
    """Infer relationships with optimized algorithm"""
    relationships = []

    # Create spatial index for faster proximity queries
    element_centers = {}

    # Calculate centers for all elements
    for text in elements['text']:
        text_center = [(text['bbox'][0][0] + text['bbox'][1][0])/2,
                      (text['bbox'][0][1] + text['bbox'][1][1])/2]
        element_centers[text['id']] = text_center

    for obj in elements['objects']:
        obj_x = sum(p[0] for p in obj['polygon']) / len(obj['polygon'])
        obj_y = sum(p[1] for p in obj['polygon']) / len(obj['polygon'])
        element_centers[obj['id']] = [obj_x, obj_y]

    # Text-object relationships based on proximity
    for text in elements['text']:
        text_id = text['id']
        text_center = element_centers[text_id]

        closest_obj = None
        min_distance = float('inf')

        for obj in elements['objects']:
            obj_id = obj['id']
            obj_center = element_centers[obj_id]

            # Calculate squared distance (faster than sqrt)
            sq_distance = (text_center[0] - obj_center[0])**2 + (text_center[1] - obj_center[1])**2

            if sq_distance < min_distance:
                min_distance = sq_distance
                closest_obj = obj

        # Use squared distance threshold (equivalent to 200 pixels)
        if closest_obj and min_distance < 40000:
            relationships.append({
                'id': f"R{len(relationships)}",
                'type': 'labels',
                'source': text_id,
                'target': closest_obj['id']
            })

    # Connector relationships
    for connector in elements['connectors']:
        start_point = connector['start']
        end_point = connector['end']

        # Find elements near start and end points
        closest_start = None
        closest_end = None
        min_start_dist = float('inf')
        min_end_dist = float('inf')

        # Check all elements
        for elem_id, center in element_centers.items():
            # Skip connectors themselves
            if elem_id.startswith('A'):
                continue

            # Calculate distances
            start_sq_dist = (start_point[0] - center[0])**2 + (start_point[1] - center[1])**2
            end_sq_dist = (end_point[0] - center[0])**2 + (end_point[1] - center[1])**2

            if start_sq_dist < min_start_dist:
                min_start_dist = start_sq_dist
                closest_start = elem_id

            if end_sq_dist < min_end_dist:
                min_end_dist = end_sq_dist
                closest_end = elem_id

        # Use squared distance threshold (equivalent to 100 pixels)
        if (closest_start and closest_end and closest_start != closest_end and
            min_start_dist < 10000 and min_end_dist < 10000):

            # Determine relationship type based on element types
            source_type = 'text' if closest_start.startswith('T') else 'blob'
            target_type = 'text' if closest_end.startswith('T') else 'blob'

            if source_type == 'text' and target_type == 'blob':
                rel_type = 'labels'
            elif source_type == 'blob' and target_type == 'blob':
                rel_type = 'connects'
            else:
                rel_type = 'relates_to'

            relationships.append({
                'id': f"R{len(relationships)}",
                'type': rel_type,
                'source': closest_start,
                'target': closest_end,
                'connector': connector['id'],
                'directional': connector.get('has_arrowhead', True)
            })

    return relationships

# Assign semantic roles
def assign_semantic_roles(elements, relationships):
    """Assign semantic roles"""
    semantic_roles = {}

    # Flatten elements
    all_elements = {}
    for elem_type, elems in elements.items():
        for elem in elems:
            all_elements[elem['id']] = elem

    # Basic role assignment
    for elem_id, elem in all_elements.items():
        if elem['type'] == 'text':
            semantic_roles[elem_id] = 'label'
        elif elem['type'] == 'blob':
            semantic_roles[elem_id] = 'object'
        elif elem['type'] == 'arrow':
            semantic_roles[elem_id] = 'connector'

    # Refine roles based on relationships
    for rel in relationships:
        source_id = rel['source']
        target_id = rel['target']

        if rel['type'] == 'labels' and source_id in all_elements and target_id in all_elements:
            if all_elements[source_id]['type'] == 'text' and all_elements[target_id]['type'] == 'blob':
                semantic_roles[source_id] = 'part_label'
                semantic_roles[target_id] = 'part'

    # Look for title patterns (text at top of image)
    texts = elements['text']
    if texts:
        sorted_texts = sorted(texts, key=lambda t: t['bbox'][0][1])
        if sorted_texts and sorted_texts[0]['bbox'][0][1] < 50:
            semantic_roles[sorted_texts[0]['id']] = 'title'

    return semantic_roles

# Classify subject
def classify_subject(elements):
    """Classify subject"""
    # Extract all text
    all_text = " ".join([elem.get('value', '').lower() for elem in elements['text']
                         if 'value' in elem])

    # Simple keyword-based classification
    subject_keywords = {
        'biology': ['cell', 'organ', 'tissue', 'plant', 'animal', 'body', 'dna', 'hair', 'eye',
                   'ear', 'nose', 'mouth', 'face'],
        'physics': ['force', 'motion', 'energy', 'mass', 'velocity', 'battery', 'magnet', 'circuit',
                   'north', 'south', 'pole'],
        'chemistry': ['element', 'compound', 'molecule', 'reaction', 'acid', 'base', 'salt',
                     'metal', 'gas', 'liquid', 'solid']
    }

    subject_scores = {}
    for subject, keywords in subject_keywords.items():
        score = sum(1 for keyword in keywords if keyword in all_text)
        subject_scores[subject] = score

    if max(subject_scores.values()) > 0:
        classified_subject = max(subject_scores.items(), key=lambda x: x[1])[0]
    else:
        classified_subject = 'general'

    # Determine diagram type
    if len(elements['connectors']) > 5:
        diagram_type = 'flow_diagram'
    elif len(elements['objects']) > len(elements['text']):
        diagram_type = 'structural_diagram'
    else:
        diagram_type = 'labeled_illustration'

    return {
        'subject': classified_subject,
        'diagram_type': diagram_type
    }

def get_explanation_from_vsrl_data(subject, key_terms, max_samples=3):
    """Generate explanation using local VSRL data instead of Gemini API"""
    global VSRL_DATA
    
    try:
        # If VSRL data not loaded yet, initialize it
        if VSRL_DATA is None:
            initialize_vsrl_data()
            
        # First try to find relevant examples by subject and keywords
        relevant_ids = []
        
        # Look for matches by subject
        subject_ids = VSRL_DATA["by_subject"].get(subject, [])
        
        # Look for matches by keywords
        keyword_matches = {}
        for term in key_terms:
            term = term.lower()
            if term in VSRL_DATA["by_keyword"]:
                for img_id in VSRL_DATA["by_keyword"][term]:
                    if img_id not in keyword_matches:
                        keyword_matches[img_id] = 0
                    keyword_matches[img_id] += 1
        
        # Sort by number of keyword matches
        sorted_matches = sorted(keyword_matches.items(), key=lambda x: x[1], reverse=True)
        
        # Prioritize examples with both subject and keyword matches
        for img_id, _ in sorted_matches:
            if img_id in subject_ids and img_id not in relevant_ids:
                relevant_ids.append(img_id)
                if len(relevant_ids) >= max_samples:
                    break
        
        # If we need more, add remaining subject matches
        if len(relevant_ids) < max_samples:
            for img_id in subject_ids:
                if img_id not in relevant_ids:
                    relevant_ids.append(img_id)
                    if len(relevant_ids) >= max_samples:
                        break
        
        # If still not enough, add keyword matches regardless of subject
        if len(relevant_ids) < max_samples:
            for img_id, _ in sorted_matches:
                if img_id not in relevant_ids:
                    relevant_ids.append(img_id)
                    if len(relevant_ids) >= max_samples:
                        break
        
        # Generate explanation from the relevant examples
        if relevant_ids:
            # Extract key information from relevant examples
            examples = []
            for img_id in relevant_ids:
                data = VSRL_DATA["by_id"][img_id]
                subject = data.get("educational_context", {}).get("subject", "")
                diagram_type = data.get("educational_context", {}).get("diagram_type", "")
                
                # Get all text entities labeled as part_label
                part_labels = []
                for entity_id, role in data.get("semantic_roles", {}).items():
                    if role == "part_label" and entity_id in data.get("entities", {}):
                        entity = data["entities"][entity_id]
                        if "value" in entity:
                            part_labels.append(entity["value"])
                
                examples.append({
                    "subject": subject,
                    "diagram_type": diagram_type,
                    "part_labels": part_labels
                })
            
            # Generate explanation text
            explanation = f"This diagram illustrates {subject} concepts. "
            
            # Mention common diagram types found
            diagram_types = [ex["diagram_type"] for ex in examples if "diagram_type" in ex]
            if diagram_types:
                most_common_type = max(set(diagram_types), key=diagram_types.count)
                explanation += f"It appears to be a {most_common_type}. "
            
            # Mention typical parts that might be labeled
            all_parts = []
            for ex in examples:
                all_parts.extend(ex.get("part_labels", []))
            
            if all_parts:
                explanation += f"In diagrams like this, important elements often include {', '.join(all_parts[:5])}. "
                explanation += f"Understanding the relationships between these components is key to mastering this {subject} concept."
            
            return explanation
        
        # Fallback to generic explanation if no relevant examples found
        return f"This diagram shows important concepts related to {subject}. Visual diagrams help students understand the relationships between different components."
            
    except Exception as e:
        print(f"Error generating explanation from VSRL data: {e}")
        return f"This diagram illustrates concepts related to {subject}."

def generate_fallback_explanation(examples_data, subject, key_terms):
    """Generate a fallback explanation if the API call fails"""
    try:
        if not examples_data:
            return f"This diagram illustrates concepts related to {subject}."
        
        # Extract common elements across examples
        all_entities = []
        for example in examples_data:
            for entity in example.get("entities", []):
                if entity.get("role") == "part_label":
                    all_entities.append(entity.get("text", "").lower())
        
        # Count frequency of entities
        entity_counts = {}
        for entity in all_entities:
            if entity:
                entity_counts[entity] = entity_counts.get(entity, 0) + 1
        
        # Sort by frequency
        sorted_entities = sorted(entity_counts.items(), key=lambda x: x[1], reverse=True)
        top_entities = [e[0] for e in sorted_entities[:5]]
        
        # Generate simple explanation
        explanation = f"This diagram illustrates {subject} concepts. "
        
        if top_entities:
            explanation += f"Key components often include {', '.join(top_entities)}. "
            explanation += f"Understanding the relationships between these elements is important for mastering {subject} concepts."
        
        return explanation
        
    except Exception as e:
        print(f"Fallback explanation error: {e}")
        return f"This diagram shows concepts related to {subject}."

def get_concept_explanation_from_gemini(subject, key_terms, max_words=100):
    """Get a concept explanation from Gemini based on VSRL training data"""
    global VSRL_DATA
    
    try:
        # Only use API if we have enough content to justify it
        if not key_terms or len(key_terms) < 2:
            return ""
            
        # Extract relevant data from VSRL dataset to provide to Gemini
        relevant_examples = []
        subject_ids = VSRL_DATA["by_subject"].get(subject, [])
        
        # Find relevant examples by keyword matching
        keyword_matches = {}
        for term in key_terms:
            term = term.lower()
            if term in VSRL_DATA["by_keyword"]:
                for img_id in VSRL_DATA["by_keyword"][term]:
                    if img_id not in keyword_matches:
                        keyword_matches[img_id] = 0
                    keyword_matches[img_id] += 1
        
        # Get top matches (prioritizing those that match both subject and keywords)
        matches_with_scores = []
        for img_id, keyword_score in keyword_matches.items():
            subject_match = 1 if img_id in subject_ids else 0
            matches_with_scores.append((img_id, keyword_score, subject_match))
        
        # Sort by subject match (primary) and keyword score (secondary)
        matches_with_scores.sort(key=lambda x: (x[2], x[1]), reverse=True)
        
        # Take top 3 examples
        top_examples = matches_with_scores[:3]
        
        # Extract data from top examples
        examples_data = []
        for img_id, _, _ in top_examples:
            if img_id in VSRL_DATA["by_id"]:
                data = VSRL_DATA["by_id"][img_id]
                
                # Get key details from this example
                subject = data.get("educational_context", {}).get("subject", "")
                diagram_type = data.get("educational_context", {}).get("diagram_type", "")
                
                # Extract text entities and their roles
                entities_info = []
                for entity_id, entity in data.get("entities", {}).items():
                    if entity.get("type") == "text" and "value" in entity:
                        role = data.get("semantic_roles", {}).get(entity_id, "unknown")
                        entities_info.append({
                            "text": entity["value"],
                            "role": role
                        })
                
                # Get relationships
                relationships = []
                for rel in data.get("relationships", []):
                    rel_type = rel.get("type", "")
                    source_id = rel.get("source", "")
                    target_id = rel.get("target", "")
                    
                    # Get source and target text if available
                    source_text = ""
                    target_text = ""
                    if source_id in data.get("entities", {}):
                        source_text = data["entities"][source_id].get("value", "")
                    if target_id in data.get("entities", {}):
                        target_text = data["entities"][target_id].get("value", "")
                    
                    if source_text and target_text:
                        relationships.append({
                            "type": rel_type,
                            "source": source_text,
                            "target": target_text
                        })
                
                examples_data.append({
                    "subject": subject,
                    "diagram_type": diagram_type,
                    "entities": entities_info,
                    "relationships": relationships
                })
        
        # Create a structured prompt with the VSRL data
        structured_data = json.dumps(examples_data, indent=2)
        
        # Create a focused prompt that asks Gemini to use only the provided data
        prompt = f"""
Based ONLY on the following educational diagram data, write a brief explanation (max {max_words} words) about {subject} concepts related to {', '.join(key_terms)}.

VSRL DATA:
{structured_data}

Your explanation should ONLY use information from the provided data. Focus on explaining the key components and their relationships as shown in similar diagrams. Do not add any information not supported by the data.
"""
        
        # Call the Gemini API
        api_key = os.getenv('GEMINI_API_KEY')  # Get API key from environment variables
        if not api_key:
            print("Warning: GEMINI_API_KEY not found in environment variables")
            return ""
            
        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={api_key}"
        
        headers = {"Content-Type": "application/json"}
        data = {
            "contents": [{
                "parts": [{"text": prompt}]
            }],
            "generationConfig": {
                "maxOutputTokens": 150,  # Limit token usage
                "temperature": 0.2       # More deterministic output
            },
            "safetySettings": [
                {
                    "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
                }
            ]
        }
        
        response = requests.post(url, headers=headers, json=data, timeout=5)
        
        if response.status_code == 200:
            result = response.json()
            if "candidates" in result and result["candidates"] and "content" in result["candidates"][0]:
                parts = result["candidates"][0]["content"].get("parts", [])
                concept_text = "".join([part.get("text", "") for part in parts])
                return concept_text
        
        # If API call fails, generate a simple explanation from the data
        return generate_fallback_explanation(examples_data, subject, key_terms)
        
    except Exception as e:
        print(f"Gemini API call error: {e}")
        return ""  # Return empty on any error

def generate_simple_explanation(subject, diagram_type, detected_texts, labeled_parts):
    """Generate a simple explanation using VSRL data and Gemini"""
    text_str = ", ".join(detected_texts)
    
    # Extract key terms for searching VSRL data
    key_terms = []
    for text in detected_texts:
        if len(text) > 3 and not text.isdigit():
            key_terms.append(text)
    
    # First get a local data-based explanation as a fallback
    local_explanation = get_explanation_from_vsrl_data(subject, key_terms)
    
    # Try to enhance with Gemini if we have enough data
    if VSRL_DATA and key_terms:
        gemini_explanation = get_concept_explanation_from_gemini(subject, key_terms, max_words=100)
        if gemini_explanation:
            # If Gemini returned something useful, use it
            vsrl_explanation = gemini_explanation
        else:
            # Otherwise fall back to local explanation
            vsrl_explanation = local_explanation
    else:
        vsrl_explanation = local_explanation
    
    # Create the explanation using the VSRL-based content
    explanation = f"""
    <h2>Educational Visual Explanation</h2>

    <p>This image is a <strong>{diagram_type}</strong> related to <strong>{subject}</strong>.</p>

    <p><strong>Content Description:</strong></p>
    <p>{vsrl_explanation}</p>
    """
    
    # Add key components section if we have labeled parts
    if labeled_parts:
        explanation += """
        <p><strong>Key Components:</strong></p>
        <ul>
        """
        for part in labeled_parts:
            explanation += f"<li><strong>{part}</strong></li>"
        explanation += "</ul>"

    # Add educational relevance based on subject
    explanation += "<p><strong>Educational Relevance:</strong></p>"

    if subject == "biology":
        explanation += """
        <p>This biological diagram helps students visualize and understand structures, processes, or relationships
        in living organisms. Visual representations like this are essential for comprehending biological concepts
        that may be difficult to observe directly.</p>
        """
    elif subject == "physics":
        explanation += """
        <p>This physics diagram helps students visualize abstract physical concepts, forces, or systems.
        Visual representations like this are essential for understanding how energy and matter interact
        in the physical world.</p>
        """
    elif subject == "chemistry":
        explanation += """
        <p>This chemistry diagram helps students visualize molecular structures, reactions, or chemical processes.
        Visual representations like this are essential for understanding the behavior of matter at scales
        that cannot be directly observed.</p>
        """
    else:
        explanation += """
        <p>This diagram helps students visualize important concepts in an accessible format.
        Visual representations like this support different learning styles and help build
        mental models of abstract concepts.</p>
        """

    # Add learning activities
    explanation += """
    <p><strong>Suggested Learning Activities:</strong></p>
    <ul>
        <li>Have students identify and describe each labeled component</li>
        <li>Ask students to explain the relationships between different elements</li>
        <li>Use the diagram as a reference point for further investigation of the topic</li>
    </ul>
    """
    
    return explanation

def visualize_vsrl(image_path, vsrl_annotation, output_dir="static/results"):
    """Visualize VSRL analysis and save visualization"""
    try:
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate a unique filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = f"{timestamp}_visualization.png"
        output_path = os.path.join(output_dir, output_filename)
        
        # Load image
        img = np.array(Image.open(image_path))
        
        # Create figure with non-interactive backend
        with plt.style.context('default'):
            fig, ax = plt.subplots(figsize=(12, 8))
            ax.imshow(img)

            # Plot entities
            for entity_id, entity in vsrl_annotation['entities'].items():
                if entity['type'] == 'text' and 'bbox' in entity:
                    x1, y1 = entity['bbox'][0]
                    x2, y2 = entity['bbox'][1]
                    width = x2 - x1
                    height = y2 - y1
                    rect = patches.Rectangle((x1, y1), width, height,
                                          linewidth=1.5, edgecolor='r', facecolor='none')
                    ax.add_patch(rect)

                    role = vsrl_annotation['semantic_roles'].get(entity_id, 'unknown')
                    ax.text(x1, y1-5, f"{entity_id}: {entity.get('value', '')}",
                          color='red', fontsize=9)

                elif entity['type'] in ['blob', 'arrow', 'connector'] and 'polygon' in entity:
                    # Convert polygon to numpy array for plotting
                    polygon = np.array(entity['polygon'])
                    if len(polygon) > 0:  # Check if polygon is not empty
                        # Make sure polygon is properly shaped for plotting
                        if len(polygon.shape) == 2 and polygon.shape[1] == 2:
                            ax.plot(polygon[:,0], polygon[:,1], 'g-', linewidth=1.5)
                        
                            if entity['type'] == 'blob':
                                # Calculate centroid for blob
                                centroid_x = np.mean(polygon[:,0])
                                centroid_y = np.mean(polygon[:,1])

                                role = vsrl_annotation['semantic_roles'].get(entity_id, 'unknown')
                                ax.text(centroid_x, centroid_y, f"{entity_id}",
                                      color='green', fontsize=9)

            # Plot relationships
            for rel in vsrl_annotation['relationships']:
                source_id = rel['source']
                target_id = rel['target']

                if source_id in vsrl_annotation['entities'] and target_id in vsrl_annotation['entities']:
                    source = vsrl_annotation['entities'][source_id]
                    target = vsrl_annotation['entities'][target_id]

                    # Calculate source coordinates
                    source_x = None
                    source_y = None
                    if 'bbox' in source:
                        source_x = (source['bbox'][0][0] + source['bbox'][1][0]) / 2
                        source_y = (source['bbox'][0][1] + source['bbox'][1][1]) / 2
                    elif 'polygon' in source and len(source['polygon']) > 0:
                        polygon = np.array(source['polygon'])
                        if len(polygon.shape) == 2 and polygon.shape[1] == 2:
                            source_x = np.mean(polygon[:,0])
                            source_y = np.mean(polygon[:,1])

                    # Calculate target coordinates
                    target_x = None
                    target_y = None
                    if 'bbox' in target:
                        target_x = (target['bbox'][0][0] + target['bbox'][1][0]) / 2
                        target_y = (target['bbox'][0][1] + target['bbox'][1][1]) / 2
                    elif 'polygon' in target and len(target['polygon']) > 0:
                        polygon = np.array(target['polygon'])
                        if len(polygon.shape) == 2 and polygon.shape[1] == 2:
                            target_x = np.mean(polygon[:,0])
                            target_y = np.mean(polygon[:,1])

                    # Draw relationship line if we have valid coordinates
                    if source_x is not None and source_y is not None and target_x is not None and target_y is not None:
                        ax.plot([source_x, target_x], [source_y, target_y], 'b--', alpha=0.6, linewidth=1)

                        # Draw relationship type at midpoint
                        mid_x = (source_x + target_x) / 2
                        mid_y = (source_y + target_y) / 2
                        ax.text(mid_x, mid_y, rel['type'], color='blue', fontsize=8,
                              bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.2'))

            # Add title
            subject = vsrl_annotation['educational_context']['subject']
            diagram_type = vsrl_annotation['educational_context']['diagram_type']
            
            plt.title(f"VSRL Analysis: {subject.capitalize()} - {diagram_type.capitalize()}")

            # Remove axes
            plt.axis('off')
            plt.tight_layout()

            # Save visualization
            plt.savefig(output_path, dpi=100, bbox_inches='tight')
            plt.close(fig)  # Explicitly close the figure
            
        return output_filename
    
    except Exception as e:
        import traceback
        print(f"Visualization error: {e}")
        print(traceback.format_exc())
        # Return a default image name if visualization fails
        return "default_visualization.png"

def flatten_elements(elements):
    """Flatten the elements dictionary for easier access"""
    entities = {}
    for elem_type, elems in elements.items():
        for elem in elems:
            entities[elem['id']] = elem
    return entities

def analyze_educational_image(image_path):
    """Analyze an educational image"""
    print(f"Analyzing image: {image_path}")

    # 1. Detect visual elements
    elements = detect_visual_elements(image_path)

    # 2. Infer relationships
    relationships = infer_relationships(elements)

    # 3. Assign semantic roles
    semantic_roles = assign_semantic_roles(elements, relationships)

    # 4. Classify subject
    subject_info = classify_subject(elements)

    # 5. Create annotation
    vsrl_annotation = {
        'image_id': os.path.basename(image_path).split('.')[0],
        'entities': flatten_elements(elements),
        'relationships': relationships,
        'semantic_roles': semantic_roles,
        'educational_context': subject_info
    }

    # 6. Visualize results
    visualization_file = visualize_vsrl(image_path, vsrl_annotation)

    # 7. Generate explanation
    detected_texts = [e.get('value', '') for e in vsrl_annotation['entities'].values()
                     if e['type'] == 'text' and 'value' in e][:10]
    
    labeled_parts = []
    for rel in relationships:
        if rel['type'] == 'labels':
            source_id = rel['source']
            if source_id in vsrl_annotation['entities']:
                source = vsrl_annotation['entities'][source_id]
                if source['type'] == 'text' and 'value' in source:
                    labeled_parts.append(source['value'])
    
    explanation = generate_simple_explanation(
        subject_info['subject'], 
        subject_info['diagram_type'],
        detected_texts,
        labeled_parts
    )

    return {
        'vsrl_annotation': vsrl_annotation,
        'visualization_file': visualization_file,
        'explanation': explanation
    }