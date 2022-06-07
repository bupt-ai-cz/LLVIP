## Annotation tutorial
### 1.download labelImg
Download the no-install version of labelImg from the link: https://github.com/tzutalin/labelImg/releases;
![labelImg](https://user-images.githubusercontent.com/33684330/172358820-df3d3871-45bb-4d3b-b36f-a2757f491638.jpeg)
Unzip the file to get the exe file of labelImg, double-click to run labelImg without installation.![labelImgexe](https://user-images.githubusercontent.com/33684330/172359515-233c3bbf-c7c5-41d4-b4a3-b637646f490b.jpeg)
### 2.use labelImg
Modify the file `predefined_classes.txt`, only keep the line 'person';![person](https://user-images.githubusercontent.com/33684330/172360612-259a1991-f371-4bc9-a619-a7f54c318513.jpeg)
Click `Open Dir` to open the image directory;

Click `Change Save Dir` to open the annotation directory;

Check the `Use default label` option and fill in `person` to speed up the labeling;![use_defult_label](https://user-images.githubusercontent.com/33684330/172363325-312b39ce-80c7-47eb-8372-4629dc4ff06f.png)

Click on `Create RectBox` on the left (shortcut key `W`), and then select the object;

![selectperson](https://user-images.githubusercontent.com/33684330/172363827-f49ac7cc-1381-45bb-bf26-0cda16ad7024.png)

Click  `Save` to save the label (shortcut key `Ctrl+S`);

Click `Next Image` to label the next image (shortcut key `D`).
