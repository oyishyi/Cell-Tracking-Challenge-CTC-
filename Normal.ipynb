{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "import cv2\n",
    "import numpy as np\n",
    "import matplotlib.pyplot as plt\n",
    "import imutils\n",
    "\n",
    "cap=cv2.VideoCapture(\"./Fluo-N2DL-HeLa/Sequence 1/t%03d.tif\")\n",
    "\n",
    "ret, frame1=cap.read()\n",
    "ret, frame2=cap.read()\n",
    "\n",
    "\n",
    "while cap.isOpened():\n",
    "    \n",
    "    if ret == True:\n",
    "        #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)\n",
    "        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))\n",
    "        img_gray = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)\n",
    "        img=frame1\n",
    "\n",
    "        ret,thresh=cv2.threshold(img_gray,129,255,cv2.THRESH_BINARY)\n",
    "        #blured = cv2.medianBlur(thresh, 7)\n",
    "        #thresh= cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)\n",
    "        #element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(7,7))\n",
    "        #b=cv2.erode(thresh,element)\n",
    "\n",
    "        blured = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)\n",
    "        #blured = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=1)\n",
    "        #blured = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel, iterations=2)\n",
    "        contours = cv2.findContours(blured,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)  \n",
    "        image = cv2.cvtColor(thresh, cv2.COLOR_GRAY2RGB)\n",
    "        #cv2.drawContours(img_gray,contours[1],-1,(0,255,0),3)\n",
    "        for contour in contours[1]:\n",
    "            (x,y,w,h) = cv2.boundingRect(contour)\n",
    "            if cv2.contourArea(contour)<40:\n",
    "                continue\n",
    "            cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)\n",
    "        cv2.putText(image, \"Number of cells: {}\".format(len(contours[1])), (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1 )\n",
    "        \n",
    "        \n",
    "        cv2.imshow('frame',image)\n",
    "        if cv2.waitKey(30) & 0xFF == ord('q'):\n",
    "            break\n",
    "        frame1 = frame2\n",
    "        ret, frame2 = cap.read()\n",
    "        \n",
    "        \n",
    "    else:\n",
    "        break\n",
    "    \n",
    "    \n",
    "cap.release()"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.7.1"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
