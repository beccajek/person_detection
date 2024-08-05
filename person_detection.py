import cv2
import numpy as np
import tflite_runtime.interpreter as tflite
from picamera2 import Picamera2
import RPi.GPIO as GPIO
import time

# Set up GPIO
GPIO.setmode(GPIO.BCM)
GPIO_PIN = 18  # Choose the GPIO pin you want to use
GPIO.setup(GPIO_PIN, GPIO.OUT)


pinIN = 17
GPIO.setup(pinIN, GPIO.IN)
pin_state = GPIO.input(pinIN)


# Load the TFLite model
interpreter = tflite.Interpreter(model_path="detect.tflite")
interpreter.allocate_tensors()

# Get input and output tensors
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Initialize camera
picam2 = Picamera2()
config = picam2.create_preview_configuration()
picam2.configure(config)
picam2.start()
#time.sleep(2)  # Give the camera time to initialize


def detect_person(image):
    # Preprocess the image
    input_shape = input_details[0]['shape']
    input_data = cv2.resize(image, (input_shape[1], input_shape[2]))
    input_data = np.expand_dims(input_data, axis=0)
 # Check the type expected by the model
    if input_details[0]['dtype'] == np.uint8:
        input_data = input_data.astype(np.uint8)
    else:
        input_data = (np.float32(input_data) - 127.5) / 127.5

    # Set the input tensor
    interpreter.set_tensor(input_details[0]['index'], input_data)

    # Run inference
    interpreter.invoke()
    
    # Get the output tensor
    boxes = interpreter.get_tensor(output_details[0]['index'])[0]
    classes = interpreter.get_tensor(output_details[1]['index'])[0]
    scores = interpreter.get_tensor(output_details[2]['index'])[0]

    # Filter for person detections (class 0 in COCO dataset)
    person_detections = [(box, score) for box, class_id, score in zip(boxes, classes, scores) 
                         if class_id == 0 and score > 0.3]

    return person_detections


try:
    while True:
        frame = picam2.capture_array()
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        detections = detect_person(frame_rgb)

        if pin_state == GPIO.HIGH: 
            print(f"door has been opened: pin 17 is HIGH")
            if detections:
                # Person detected, set GPIO pin high
                GPIO.output(GPIO_PIN, GPIO.LOW)
                GPIO.setup(pinIN, GPIO.OUT)
                GPIO.output(pinIN, GPIO.LOW)
                pin_state = GPIO.input(pinIN)

                print("Person detected! pin 17 set LOW")
                print(f"Read state: {'HIGH' if pin_state else 'LOW'}")

            else:
                # No person detected, set GPIO pin low
                GPIO.output(GPIO_PIN, GPIO.HIGH)
                print("No person detected. GPIO pin set HIGH")
                
         
        if pin_state == GPIO.LOW:
            if detections:
                # Person detected, set GPIO pin low
                GPIO.output(GPIO_PIN, GPIO.LOW)
                print(f"Read state: {'HIGH' if pin_state else 'LOW'}")
                print("Person still detected! GPIO pin continues LOW")
            else:
                # No person detected, set GPIO pin low
                GPIO.output(GPIO_PIN, GPIO.HIGH)
                print("person cleared. GPIO pin set HIGH")
                GPIO.setup(pinIN, GPIO.OUT)
                GPIO.output(pinIN, GPIO.HIGH)
                pin_state = GPIO.input(pinIN)
                print(f"Read state: {'HIGH' if pin_state else 'LOW'}")

             
             
        # Draw bounding boxes for detected persons
        for box, score in detections:
            ymin, xmin, ymax, xmax = box
            xmin = int(xmin * frame.shape[1])
            xmax = int(xmax * frame.shape[1])
            ymin = int(ymin * frame.shape[0])
            ymax = int(ymax * frame.shape[0])
            
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
            cv2.putText(frame, f"Person: {score:.2f}", (xmin, ymin - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        cv2.imshow("Person Detection", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    cv2.destroyAllWindows()
    picam2.stop()
    GPIO.cleanup()  # Clean up GPIO on exit
