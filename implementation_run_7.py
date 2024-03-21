import pickle
import cv2
import mediapipe as mp
import numpy as np
import time
 
liy=[]

model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']


labels_dict = {0: 'zero', 1: 'one', 2: 'two', 3: 'three', 4: 'four', 5: 'five', 6: 'ok', 7: 'back'}


cap = cv2.VideoCapture(0)


mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)


input_buffer = []
input_debounce = False
debounce_time = 1  


labels_to_digits = {
    'one': '1',
    'two': '2',
    'three': '3',
    'four': '4',
    'five': '5',
    'zero': '0'  
}


def visualize_path(floor_number):

    print(f"Visualizing path for floor number: {floor_number}")


while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    H, W, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks and not input_debounce:
        for hand_landmarks in results.multi_hand_landmarks:
           
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )

            
            x_ = [landmark.x for landmark in hand_landmarks.landmark]
            y_ = [landmark.y for landmark in hand_landmarks.landmark]
            x1, y1, x2, y2 = int(min(x_) * W), int(min(y_) * H), int(max(x_) * W), int(max(y_) * H)

           
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

            
            data_aux = [(landmark.x - min(x_), landmark.y - min(y_)) for landmark in hand_landmarks.landmark]
            data_aux_flat = [item for sublist in data_aux for item in sublist]

           
            prediction = model.predict([data_aux_flat])
            predicted_character = labels_dict[int(prediction[0])]

            
            cv2.putText(frame, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            
            if predicted_character in labels_to_digits:
                if len(input_buffer) < 3:
                    input_buffer.append(labels_to_digits[predicted_character])
                    input_debounce = True  
                    debounce_end_time = time.time() + debounce_time
            elif predicted_character == 'back':
                if input_buffer:
                    input_buffer.pop()
                    input_debounce = True  
                    debounce_end_time = time.time() + debounce_time
            elif predicted_character == 'ok':
                if input_buffer:
                    floor_number = ''.join(input_buffer)
                    visualize_path(floor_number)
                    liy = input_buffer  
                    input_debounce = True  
                    debounce_end_time = time.time() + debounce_time

    
    if input_debounce and time.time() >= debounce_end_time:
        input_debounce = False  

    
    input_display = "Floor: " + ''.join(input_buffer).ljust(3, '_')
    cv2.putText(frame, input_display, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

    
    cv2.imshow('frame', frame)

  
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()
lily = ''.join(liy)



start_location = (1215, 516) 
room_locations = {
    '101': (340, 358), '102': (550, 364), '103': (340, 643), '104': (564, 649), '105': (1023, 644),  # 1st floor
    '201': (340, 358), '202': (550, 364), '203': (340, 643), '204': (564, 649), '205': (1023, 644),  # 2nd floor
    '301': (340, 358), '302': (550, 364), '303': (340, 643), '304': (564, 649), '305': (1023, 644),  # 3rd floor
    '401': (340, 358), '402': (550, 364), '403': (340, 643), '404': (564, 649), '405': (1023, 644),  # 4th floor
    '501': (340, 358), '502': (550, 364), '503': (340, 643), '504': (564, 649), '505': (1023, 644),  # 5th floor
}



waypoints = {
    '1': [(1185, 553), (1021, 553), (553, 553)],  
    '2': [(1185, 553), (1021, 553), (553, 553)],
    '3': [(1185, 553), (1021, 553), (553, 553)],
    '4': [(1185, 553), (1021, 553), (553, 553)],
    '5': [(1185, 553), (1021, 553), (553, 553)],  
    
}


def load_floor_plan(floor_number):
    image_path = f"floor-plan/{floor_number}.png"  
    floor_plan = cv2.imread(image_path)
    if floor_plan is None:
        raise FileNotFoundError(f"No floor plan found for floor number {floor_number}")
    return floor_plan


def draw_path(floor_plan, path):
    
    for i in range(len(path) - 1):
        cv2.line(floor_plan, path[i], path[i+1], (255,0, 0), 2)  


def show_path_to_room(room_number):
    
    floor_number = room_number[0]
    floor_plan = load_floor_plan(floor_number)

    
    path_to_room = [start_location] + waypoints[floor_number] + [room_locations[room_number]]

    
    draw_path(floor_plan, path_to_room)

    
    cv2.imshow(f'Path to Room {room_number}', floor_plan)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


show_path_to_room(lily)
