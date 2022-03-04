import cv2

# crops hand portion in the video.
# Resize each frame : 200 by 200
# Adds every frame of each video to 'data' folder which will be later used to create dataset.

class HandCropper():

    def __init__(self, xMin, xMax, yMin, yMax, inVideoPath, outVideoPath, outDataFolder, initialFrameCount):
        self.xMin = xMin
        self.xMax = xMax
        self.yMin = yMin
        self.yMax = yMax
        self.inVideoPath = inVideoPath
        self.outVideoPath = outVideoPath
        self.outDataFolder = outDataFolder
        self.initialFrameCount = initialFrameCount
        
    
    def cropHands(self):
        input_video = cv2.VideoCapture(self.inVideoPath)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        output_video = cv2.VideoWriter(self.outVideoPath, fourcc, 30, (200, 200))
        # output_video = cv2.VideoWriter(self.outVideoFolder, fourcc, 30, (min(self.xMax+30,1080)-max(self.xMin-40,0),min(self.yMax+30,1920)-max(self.yMin-30,0)))
        while True:
            ret, frame = input_video.read()
            if not ret:
                break
            
            # Cropping each frame as per the hand coordinates
            crop_img = frame[max(self.yMin-30,0):min(self.yMax+30,1920), max(self.xMin-40,0):min(self.xMax+30,1080), :]
            cropped_resized_img = cv2.resize(crop_img, (200, 200))
            output_video.write(cropped_resized_img)

            # Writing cropped frame to data folder
            cv2.imwrite("%s/%d.jpg"%(self.outDataFolder,self.initialFrameCount),cropped_resized_img)
            self.initialFrameCount = self.initialFrameCount + 1

        
        output_video.release()
        return self.initialFrameCount


        