"""L'objectif de ces expériences est de savoir comment se comporte TextureSam sous prompting
Code en majorité issu de https://github.com/facebookresearch/sam2/blob/main/notebooks/image_predictor_example.ipynb

"""
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
from sam2.build_sam import build_sam2
from  sam2.sam2_image_predictor import SAM2ImagePredictor

from sam2. automatic_mask_generator import SAM2AutomaticMaskGenerator








class SamPredictor : 


    def __init__(self,model_cfg,checkpoint):


        self.device="cuda"
        self.current_mask_generator="Promptable" ##ne peut prendre que 2 valeurs : "Promptable","Automatic"
     
        self.sam2 = build_sam2(model_cfg, checkpoint, device=self.device, apply_postprocessing=False)
    
        self.predictor = SAM2ImagePredictor(

            sam_model=self.sam2,
            points_per_side=64,
            pred_iou_thresh=0.8,
            stability_score_thresh=0.2,
            mask_threshold=0.0,
            min_mask_region_area=0,
            output_mode="binary_mask",
            multimask_output=True
        
        )
        


    def __change_predictor_type(self): 


        if self.current_mask_generator=="Promptable" : 

            self.current_mask_generator="Automatic"
            self.predictor= SAM2AutomaticMaskGenerator(
                model=self.sam2,
                points_per_side=64,
                pred_iou_thresh=0.8,
                stability_score_thresh=0.2,
                mask_threshold=0.0,
                min_mask_region_area=0,
                output_mode="binary_mask",
                multimask_output=True
            )
        elif   self.current_mask_generator=="Automatic" : 
            
            
            self.current_mask_generator="Promptable" 

            self.predictor = SAM2ImagePredictor(

                            sam_model=self.sam2,
                            points_per_side=64,
                            pred_iou_thresh=0.8,
                            stability_score_thresh=0.2,
                            mask_threshold=0.0,
                            min_mask_region_area=0,
                            output_mode="binary_mask",
                            multimask_output=True
                        
                        
                        )
            
    def predict(self, image:np.ndarray, points_coord:list=None,points_labels:list=None) : 


        
        if len(points_coord)!=0 and len(points_labels)!=0 : 
            

            if self.current_mask_generator=="Automatic" : 
                    self.__change_predictor_type()
            self.predictor.set_image(image)
           
            masks,score,_=self.predictor.predict(point_coords=points_coord,point_labels=points_labels)
           
        else : 

            if self.current_mask_generator=="Promptable" :
                
                 self.__change_predictor_type()
            
         
            masks=self.predictor.generate(image)
            score=None
        
        return masks,score


                                               
        


            

class Imageprocessor: 

    def __init__(self, SamPredictor : SamPredictor) : 

        self.predictor=SamPredictor
 

    

    def show_mask(self,mask, ax, random_color=False, borders = True):
        if random_color:
            color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
        else:
            color = np.array([30/255, 144/255, 255/255, 0.6])
        h, w = mask.shape[-2:]
        mask = mask.astype(np.uint8)
        mask_image =  mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
        if borders:
            import cv2
            contours, _ = cv2.findContours(mask,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) 
            # Try to smooth contours
            contours = [cv2.approxPolyDP(contour, epsilon=0.01, closed=True) for contour in contours]
            mask_image = cv2.drawContours(mask_image, contours, -1, (1, 1, 1, 0.5), thickness=2) 
        ax.imshow(mask_image)


    def show_points(self,coords, labels, ax, marker_size=375):
        pos_points = coords[labels==1]
        neg_points = coords[labels==0]
        print("ax",pos_points)
        ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
        ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   


    def show_box(self,box, ax):
        x0, y0 = box[0], box[1]
        w, h = box[2] - box[0], box[3] - box[1]
        ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))    

    def show_masks(self,image, masks, scores, point_coords=None, box_coords=None, input_labels=None, borders=True):
        for i, (mask, score) in enumerate(zip(masks, scores)):
            plt.figure(figsize=(10, 10))
            plt.imshow(image)
            self.show_mask(mask, plt.gca(), borders=borders)
            if point_coords is not None:
                assert input_labels is not None
                self.show_points(point_coords, input_labels, plt.gca())
            if box_coords is not None:
                # boxes
                self.show_box(box_coords, plt.gca())
            if len(scores) >= 1:
                plt.title(f"Mask {i+1}, Score: {score}", fontsize=18)
            plt.axis('off')
            plt.show()









    def process(self, dic_image :dict) : 


        """
            Il faut un dictionnaire ayant pour structure : 

            {"image path" : str, 
            "points_coords": list[list[tuples(int,int]],
            "points_labels" : list[list[int]]                        
                                    
             }

             évidemment, il faut que les longueurs de listes primaires/secondaires correspondent entre coords et labels 

        """
            
        assert len(dic_image["points_coords"])==len(dic_image["points_labels"])

        image = Image.open(dic_image["image_path"]).convert("RGB")
        image_array = np.array(image)

        for idx,prompt in enumerate(dic_image["points_coords"]):
            
            
            prompt=np.array([list(i) for i in prompt]) ## juste du formatage pour passer correctement 
            print("prompt ! ", prompt)
           
            assert len(prompt)==len(dic_image["points_labels"][idx])
            if len(prompt)==0 : 
                


                masks,score=self.predictor.predict(image=image_array,points_coord=prompt,points_labels=dic_image["points_labels"][idx])
                print("len mask !!!", len(masks))
                scores=np.array([i["stability_score"] for i in masks])
                masks=np.array([i["segmentation"].astype(np.int8) for i in masks])
                print("scores !!!",scores)
                self.show_masks(image_array,masks,scores,point_coords=None,input_labels=None)
              

            else: 
                    
                    masks,score=self.predictor.predict(image=image_array,points_coord=prompt,points_labels=dic_image["points_labels"][idx])
                    print(masks)
                    print("mask", type(masks),"score",type(score))
                    labels=np.array(dic_image["points_labels"][idx])
                    self.show_masks(image_array,masks,score,point_coords=prompt,input_labels=labels)

            
            













if __name__=="__main__" :


    image_dir = "./input_images" 
    output_dir = "./segmented_images" 
    model_cfg="//home/abouchet/Documents/Datasets/Test_textureSam/configs/sam2.1_hiera_s.yaml"
    checkpoint="/home/abouchet/Documents/Datasets/Test_textureSam/checkpoints/sam2.1_hiera_small_1.pt"

    P1=SamPredictor(model_cfg=model_cfg,checkpoint=checkpoint)
    IP=Imageprocessor(P1)

    
    dic_={"image_path" : "/home/abouchet/Documents/Datasets/Test_DataSet_LatentClustering_2/images/060722-Nabila-JP-Valves-WholeMount-SAureus-pat04-1-37.tif", 
            "points_coords": [[],[[920,688],[969,504],[1074,636]],[[920,688],[954,196]],[[920,688]],[[400,300],[920,688],[954,196]]],
            "points_labels" : [[],[1,1,1],[1,0],[1],[1,1,0]]                                     
             }
    dic_={"image_path" : "/home/abouchet/Documents/Datasets/Test_textureSam/MEB/input_images/060525-JPB-MEB-EIHNValves-Ech2-ZigZag0037.tiff", 
            "points_coords": [[],[[700,72]],[[700,72],[773,600]],[[700,72],[194,200],[1100,235],[773,600],[191,670],[1130,470]]],
            "points_labels" : [[],[1],[1,0],[1,1,1,0,0,0]],                                  
             }
   
     
    dic_={"image_path" : "/home/abouchet/Documents/Datasets/Test_textureSam/MEB/input_images/060525-JPB-MEB-EIHNValves-Ech2-ZigZag0071.tiff", 
            "points_coords": [[], [[200,130],[600,200],[900,200],[1230,180]],[[50,290],[200,130],[600,200],[900,200],[1230,180],[250,635],[780,420],[950,600],[115,630],[244,461]]],
            "points_labels" : [[],[1,1,1,1],[1,1,1,1,1,0,0,0,0,0]]                                     
             }
    dic_={"image_path" : "/home/abouchet/Documents/Datasets/Test_textureSam/MEB/input_images/070525-JPB-MEB-EIHNValves-Ech4-ZigZag0069.tiff", 
            "points_coords": [[],[[822,700],[922,556],[992,384],[1180,574],[1180,750]],[[822,700],[922,556],[992,384],[1180,574],[1180,750],[650,34],[822,138],[938,76]],[[822,700],[922,556],[992,384],[1180,574],[1180,750],[650,34],[822,138],[938,76],[1060,216],[1202,296],[1210,166],[1124,26],[756,369],[310,150],[216,468],[102,672]]],
            "points_labels" : [[],[1,1,1,1,1],[1,1,1,1,1,1,1,1],[1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0]]                                     
             }



  
   
    IP.process(dic_)




