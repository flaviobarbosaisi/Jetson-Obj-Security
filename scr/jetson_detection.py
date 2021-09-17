"""Author: Flavio Barbosa"""


import os
import warnings
import cv2
import numpy as np
import plot
import imutils
import time



warnings.filterwarnings('ignore')


class LargeObjects:
    """class to monitor the approach of people to a particular object"""
    def __init__(self, video_path):
        #self.init_path = None
        self.video_path = video_path
        self.model = None
        self.calibra = 0
        self.mouse_pts = []
        self.frame = None
        self.frame_cp = None
        self.rects = []


    def load_models(self, init_path ):
        """function to load models. It has as input the configuration files (cfg and weights) and the input size"""
        self.init_path = init_path
        cfg_path_placa = init_path+"yolov4.cfg"
        weights_path_placa = init_path+"yolov4_last.weights"
        input_size_placa = (416, 416)
        self.model = self.load_yolo_detectors(cfg_path_placa, weights_path_placa,
                                              input_size_placa)

    @staticmethod
    def load_yolo_detectors(cfg, weights, input_size):
        """This method uses opencv to load the YOLOv4 detector. It also configures to preferentially use gpu if possible"""
        net = cv2.dnn_DetectionModel(cfg, weights)
        net.setInputSize(input_size)
        net.setInputScale(1.0 / 255)
        net.setInputSwapRB(True)
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
        return net


    def get_mouse_points(self, event, x, y, flags, param):

        """To calibrate the system, the user must use the mouse to indicate 7 points.
        #The first 4 points or coordinates for perspective transformation.
        #The region marked by these 4 points are considered ROI. The remaining 3 are used
        # for for horizontal and vertical unit length calibration. This function recovers
        #the user's mouse clicks."""
        image = self.frame
        if event == cv2.EVENT_LBUTTONDOWN:
            if len(self.mouse_pts) < 4:
                cv2.circle(image, (x, y), 5, (0, 0, 255), 10)
            else:
                cv2.circle(image, (x, y), 5, (255, 0, 0), 10)

            if len(self.mouse_pts) >= 1 and len(self.mouse_pts) <= 3:
                cv2.line(image, (x, y), (self.mouse_pts[len(self.mouse_pts)-1][0],
                                         self.mouse_pts[len(self.mouse_pts)-1][1]), (70, 70, 70), 2)
                if len(self.mouse_pts) == 3:
                    cv2.line(image, (x, y), (self.mouse_pts[0][0],
                                             self.mouse_pts[0][1]), (70, 70, 70), 2)

            self.mouse_pts.append((x, y))



    def calibra_points(self):
        """It uses the first frame as an input for calibration"""
        cv2.namedWindow("image")
        cv2.setMouseCallback("image", self.get_mouse_points)
        while True:
            image = self.frame
            cv2.imshow("image", image)
            cv2.waitKey(1)
            if len(self.mouse_pts) == 8:
                cv2.destroyWindow("image")
                break

        self.calibra = 1
        return self.mouse_pts

    @staticmethod
    def homography(points, height, width):
        """
        # Using first 4 points or coordinates for perspective transformation.
        #The region marked by these 4 points are considered ROI. This polygon
        #shaped ROI is then warped into a rectangle which becomes the bird eye view.
        # This bird eye view then has the property property that points are distributed
        #uniformally horizontally and vertically(scale for horizontal and vertical
        #direction will be different). So for bird eye view points are
        # equally distributed, which was not case for normal view."""
        src = np.float32(np.array(points[:4]))
        dst = np.float32([[0, height], [width, height], [width, 0], [0, 0]])
        prespective_transform = cv2.getPerspectiveTransform(src, dst)

        # using next 3 points for horizontal and vertical unit length(in this case 170 cm)
        pts = np.float32(np.array([points[4:7]]))
        warped_pt = cv2.perspectiveTransform(pts, prespective_transform)[0]

        # Using first 4 points or coordinates for perspective transformation.
        # The region marked by these 4 points are considered ROI. This
        # polygon shaped ROI is then warped into a rectangle which becomes
        # the bird eye view. This bird eye view then has the property
        # that points are distributed uniformally horizontally and
        # vertically(scale for horizontal and vertical direction will be different).
        #So for bird eye view points are
        # equally distributed, which was not case for normal view.
        src = np.float32(np.array(points[:4]))
        dst = np.float32([[0, height], [width, height], [width, 0], [0, 0]])
        prespective_transform = cv2.getPerspectiveTransform(src, dst)

        # using next 4 points for horizontal and vertical unit length(in this case 170 cm)
        pts = np.float32(np.array([points[4:7]]))
        warped_pt = cv2.perspectiveTransform(pts, prespective_transform)[0]

        # since bird eye view has property that all points are equidistant
        #in horizontal and vertical direction. distance_w and distance_h
        #will give us 170 cm distance in both horizontal and vertical
        #directions (how many pixels will be there in 170cm length in
        #horizontal and vertical direction of birds eye view),
        # which we can use to calculate distance between two humans
        # in transformed view or bird eye view
        distance_w = np.sqrt((warped_pt[0][0] - warped_pt[1][0]) ** 2 +
                             (warped_pt[0][1] - warped_pt[1][1]) ** 2)
        distance_h = np.sqrt((warped_pt[0][0] - warped_pt[2][0]) ** 2 +
                             (warped_pt[0][1] - warped_pt[2][1]) ** 2)

        return distance_w, distance_h, prespective_transform

    def detect_person_object(self, image):
        """The trained YOLOv4 detector is used to detect the people and
        #the object of interest present in the scene"""
        boxes_p = []
        confidences_p = []
        boxes_o = []
        confidences_o = []
        allboxes = []
        self.rects = []

        classes, confidences, boxes = self.model.detect(image, confThreshold=0.5, nmsThreshold=0.1)
        for i in range(0, len(confidences)):
            if confidences[i] > 0.3:
                box = boxes[i]
                (start_x, start_y, width, height) = box.astype("int")
                x = start_x
                y = start_y

                (startX, startY, sizeX, sizeY) = boxes[i].astype("int")
                endX = startX + sizeX
                endY = startY + sizeY
                box_rect = startX, startY, endX, endY
                self.rects.append(box_rect)

                ### Person
                if classes[i] == 0:
                    boxes_p.append([x, y, int(width), int(height)])
                    confidences_p.append(float(confidences[i]))
                    allboxes.append([x, y, int(width), int(height)])
                ###object
                elif classes[i] == 1:
                    boxes_o.append([x, y, int(width), int(height)])
                    confidences_o.append(float(confidences[i]))
                    allboxes.append([x, y, int(width), int(height)])

                cv2.rectangle(self.frame_cp, (startX, startY), (endX, endY),
            (0, 255, 0), 2)


        return boxes_p, boxes_o, allboxes

    @staticmethod
    def get_transformed_points(boxes, prespective_transform):
        """To better calculate the distance between people and the object of interest,
        #I used the lower midpoint of the bounding box after bird's eye view transformation"""
        bottom_points = []
        for box in boxes:
            pnts = np.array([[[int(box[0]+(box[2]*0.5)), int(box[1]+box[3])]]], dtype="float32")
            bd_pnt = cv2.perspectiveTransform(pnts, prespective_transform)[0][0]
            pnt = [int(bd_pnt[0]), int(bd_pnt[1])]
            bottom_points.append(pnt)

        return bottom_points

    @staticmethod
    def cal_dis(point1, point2, distance_w, distance_h):
        """Through the transformed points, the approximate distance between the person
        and the object of interest is calculated"""
        height = abs(point2[1]-point1[1])
        width = abs(point2[0]-point1[0])

        dis_w = float((width/distance_w)*170)
        dis_h = float((height/distance_h)*170)

        return int(np.sqrt(((dis_h)**2) + ((dis_w)**2)))



    def get_distances2(self, boxes1, boxes2, object_points, person_points, distance_w, distance_h):
        """The calculated distances allow assigning risk levels according to proximity.
        If the person approaches 100 cm or less from the object, the high risk level is attributed,
         between 100 and 125 cm, medium risk and above this distance,
         it was considered that there are no risks."""
        distance_mat = []
        bxs = []


        for i in range(len(person_points)):
            flag = 0
            for j in range(len(object_points)):
                dist = self.cal_dis(person_points[i], object_points[j], distance_w, distance_h)
                if dist <= 100 and flag != 1:
                    closeness = 0
                    flag = 1
                elif dist > 100 and dist <= 125 and flag != 2 and flag != 1:
                    closeness = 1
                    flag = 2
                elif dist > 125 and flag == 0:
                    closeness = 2
                    flag = 3
            if len(object_points) > 0:
                distance_mat.append([person_points[i], object_points[j], closeness])
                bxs.append([boxes1[i], boxes2[j], closeness])

        return distance_mat, bxs



    @staticmethod
    def get_count2(distances_mat, boxes_p, boxes_o):
        """method to count how many people are at which risk levels"""
        high = 0
        medium = 0
        low = 0

        if len(boxes_o) == 0:
            low = len(boxes_p)
        else:
            for i in range(len(distances_mat)):

                if distances_mat[i][2] == 0:
                    high += 1
                if distances_mat[i][2] == 1:
                    medium += 1
                if distances_mat[i][2] == 2:
                    low += 1
        return (low, medium, high)



    def risk_analysys(self, boxes_p, boxes_o,
                      distance_w, distance_h, perspective_transform):

        """This function perform the risk analysis based on the transformed points"""
        person_points = self.get_transformed_points(boxes_p, perspective_transform)
        object_points = self.get_transformed_points(boxes_o, perspective_transform)

        # Here we will calculate distance between transformed points(humans and object)


        distances_mat, bxs_mat = self.get_distances2(boxes_p, boxes_o, object_points,
                                              person_points, distance_w, distance_h)


        #count the number of people in each risk level
        risk_count2 = self.get_count2(distances_mat, boxes_p, boxes_o)

        num_lr, num_mr, num_hr = risk_count2


        if ((len(boxes_o) < 1) or (len(boxes_p) < 1)):
            num_lr, num_mr, num_hr = [0, 0, 0]



        return num_lr, num_mr, num_hr, distances_mat, bxs_mat, risk_count2

       # return float(dis_w/width), float(dis_h/height)


    def run(self):
        """Function to organize the code flow"""

        #if camera/webcam
        #cap = cv2.VideoCapture(0)
        #if video
        cap = cv2.VideoCapture(FILE_PATH)
        #
        time.sleep(0.1)


        #fourcc = cv2.VideoWriter_fourcc(*'mp4v')

        #Count the bumber of frames received and variable to detect each X frames. Using Jetson Xavier, it is possible
        #to run in real time without skipping frames
        skip_frames = 1

        total_frames = 0

        #to record videos
        #writer = None

        # Process each frame, until end of video
        while cap.isOpened():
            ret, self.frame = cap.read()



            if not ret:
                print("end of the video file...")
                break

            self.frame = imutils.resize(self.frame, width=800)

            self.frame_cp = self.frame.copy()

            height = self.frame.shape[0]
            width = self.frame.shape[1]


            #check if the system is calibrated, it is mandatory a calibration before usage
            if self.calibra == 0:
                points = self.calibra_points()

            #calculates homography for the Bird's Eye View Transform
            dist_w, dist_h, perspective = self.homography(points, height, width)

            #The trained YOLOv4 is used for detecting the object of interest and people in the ROI
            if total_frames % skip_frames == 0:
                boxes_p, boxes_o, allboxes = self.detect_person_object(self.frame)


            #An analysis is perfomed to check if there is someone near the object of interest
            num_lr, num_mr, num_hr, distances_mat, bxs_mat, risk_count2 = self.risk_analysys(boxes_p, boxes_o,
                                                        dist_w, dist_h,
                                                        perspective)

            risk_plot =  num_hr, num_mr, num_lr

            frame1 = np.copy(self.frame)

            #Plots on the screan the distance analysis result
            img = plot.distancing_view(frame1, bxs_mat, allboxes, risk_plot)
            # parameters for video recording
            fshape = img.shape
            height = fshape[0]
            width = fshape[1]
            fps = int(cap.get(cv2.CAP_PROP_FPS))

            #if writer == None:
            #    writer = cv2.VideoWriter("path/distancing_jetson_online22222.mp4", fourcc, fps, (width, height))

            #writer.write(img)

            #Frame counting
            total_frames += 1

            cv2.imshow("Frame", img)


            if cv2.waitKey(1) & 0xFF == ord('q'):
                break


        writer.release()
        cap.release()
        cv2.destroyAllWindows()
        print ("finishing...")



if __name__ == '__main__':


    #FILE_PATH = "path/to/Jetson-Obj-Security/data/sample_test/sample.mp4" #for video
    #FILE_PATH = 0 #for webcam


    RISK_ANALYSYS = LargeObjects(FILE_PATH)


   # SUB_PATH = "path/to/Jetson-Obj-Security/models/"
    RISK_ANALYSYS.load_models(SUB_PATH)

    RISK_ANALYSYS.run()
