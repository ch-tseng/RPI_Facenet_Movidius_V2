import time
import imutils
import cv2
import numpy as np
#from mvnc import mvncapi as mvnc
import mvnc_simple_api as mvnc
import paho.mqtt.client as mqtt

class webCam:
    def __init__(self, id, size=(320, 240)):
        self.cam = cv2.VideoCapture(id)
        self.cam.set(cv2.CAP_PROP_FRAME_WIDTH, size[0])
        self.cam.set(cv2.CAP_PROP_FRAME_HEIGHT, size[1])

    def working(self):
        webCam = self.cam
        if(webCam.isOpened() is True):
            return True
        else:
            return False

    def takepic(self, rotate=0, vflip=False, hflip=False, resize=None, savePath=None):
        webcam = self.cam
        hasFrame, frame = webcam.read()

        if(vflip==True):
            frame = cv2.flip(frame, 0)
        if(hflip==True):
            frame = cv2.flip(frame, 1)

        if(rotate>0):
            frame = imutils.rotate(frame, rotate)
        if(resize is not None):
            frame = imutils.resize(frame, size=resize)
        if((hasFrame is True) and (savePath is not None)):
            cv2.imwrite(savePath+str(time.time())+".jpg", frame)

        return hasFrame, frame

    def release(self):
        webcam = self.cam
        webcam.release()

class facenetVerify:
    def __init__(self, graphPath, movidiusID=0):
        devices = mvnc.EnumerateDevices()
        if len(devices) == 0:
            logging.critical('No NCS devices found')
            quit()

        # Pick the first stick to run the network
        device = mvnc.Device(devices[movidiusID])

        # Open the NCS
        device.OpenDevice()

        # read in the graph file to memory buffer
        with open(graphPath, mode='rb') as f:
            graph_in_memory = f.read()

            # create the NCAPI graph instance from the memory buffer containing the graph file.
            self.graph = device.AllocateGraph(graph_in_memory)

    def __whiten_image(self, source_image):
        source_mean = np.mean(source_image)
        source_standard_deviation = np.std(source_image)
        std_adjusted = np.maximum(source_standard_deviation, 1.0 / np.sqrt(source_image.size))
        whitened_image = np.multiply(np.subtract(source_image, source_mean), 1 / std_adjusted)
        return whitened_image


    def __preprocess_image(self, src):
        # scale the image
        NETWORK_WIDTH = 160
        NETWORK_HEIGHT = 160
        preprocessed_image = cv2.resize(src, (NETWORK_WIDTH, NETWORK_HEIGHT))

        #convert to RGB
        preprocessed_image = cv2.cvtColor(preprocessed_image, cv2.COLOR_BGR2RGB)

        #whiten
        preprocessed_image = self.__whiten_image(preprocessed_image)

        # return the preprocessed image
        return preprocessed_image


    def __run_inference(self, image_to_classify, facenet_graph):

        # get a resized version of the image that is the dimensions
        # SSD Mobile net expects
        resized_image = self.__preprocess_image(image_to_classify)

        # ***************************************************************
        # Send the image to the NCS
        # ***************************************************************
        facenet_graph.LoadTensor(resized_image.astype(np.float16), None)

        # ***************************************************************
        # Get the result from the NCS
        # ***************************************************************
        output, userobj = facenet_graph.GetResult()

        #print("Total results: " + str(len(output)))
        #print(output)

        return output

    def load_dataset(self, dsPath):
        hf = h5py.File(dataset_file, 'r')
        valid_uids = hf.get('uids')
        valid_embs = hf.get('embs')

        print("HF file loaded, valid names:", valid_uids)
        return (valid_uids, valid_embs)


    def make_dataset(self, facesPath, outputPath="dataset_embs", camlist=["cam0", "cam1"]):
        #dataset format: 1 camera 1 h5 file, so we have cam0 and cam1 ,2 h5 files

        valid_uid = []
        valid_embs = []
        for CAM in camlist:
            hf = h5py.File(outputPath+"_"+CAM, 'w')
            for UID in os.listdir(facesPath):  # User id list
                for PHOTO in os.listdir(facesPath + "/" + UID + "/" + CAM):  #photo list
                    folder = facesPath + "/" + UID + "/" + CAM
                    print("Processing ", folder + "/" + PHOTO)
                    filename, file_extension = os.path.splitext(PHOTO)
                    file_extension = file_extension.lower()
                    if(file_extension == ".jpg" or file_extension==".jpeg" or file_extension==".png" or file_extension==".bmp"):
                        face = cv2.imread(folder + "/" + PHOTO)
                        embs_face = self.__run_inference(face, self.graph)
                        valid_uid.append(UID.encode())
                        valid_embs.append(embs_face)

        if(len(valid_embs)>0 and len(valid_uid)>0):
            hf.create_dataset("uids", data=np.array(valid_uid))
            hf.create_dataset("embs", data=np.array(valid_embs))
            hf.close()

        print("EMBS DATASET created.")

    def face_match(self, face1, face2, threshold):
        face1_output = self.__run_inference(face1, self.graph)
        face2_output = self.__run_inference(face2, self.graph)

        if (len(face1_output) != len(face2_output)):
            logging.error('Face for facenet data length mismatch in face_match')
            return False


        total_diff = 0
        for output_index in range(0, len(face1_output)):
            this_diff = np.square(face1_output[output_index] - face2_output[output_index])
            total_diff += this_diff
        #print('difference is: {}'.format(total_diff))

        if (total_diff < threshold):
            # the total difference between the two is under the threshold so
            # the faces match.
            #print('Pass! difference is: ' + str(total_diff))
            return True, total_diff
        else:
            # differences between faces was over the threshold above so
            # they didn't match.
            #print('No pass! difference is: ' + str(total_diff))
            return False, total_diff

class mqttFACE():
    def __init__(self, host, chnl, portnum):
        self.host = host
        self.channel = chnl
        self.port = portnum

    def sendMQTT(self, msg):
        mqttc = mqtt.Client("Face-Checkin")
        mqttc.username_pw_set("chtseng", "chtseng")
        mqttc.connect(self.host, self.port)
        mqttc.publish(self.channel, msg)
