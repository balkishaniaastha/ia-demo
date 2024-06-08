import cv2
import os
import layoutparser as lp
import pdf2image
import numpy as np
import pytesseract

class Box_detection:
    def __init__(self, model_config='lp://TableBank/faster_rcnn_R_101_FPN_3x/config', score_thresh=0.1, device='cpu',
                 label_map={0: "Table"}):
        self.model_weights_path = os.path.expanduser('~/.torch/iopath_cache/s/6vzfk8lk9xvyitg/model_final.pth')
        if not os.path.isfile(self.model_weights_path):
            raise FileNotFoundError(
                f"Model weights not found at {self.model_weights_path}. Please download the weights manually.")

        self.model = lp.Detectron2LayoutModel(model_config, extra_config=["MODEL.WEIGHTS", self.model_weights_path,
                                                                          "MODEL.ROI_HEADS.SCORE_THRESH_TEST",
                                                                          score_thresh, "MODEL.DEVICE", device],
                                              label_map=label_map)

    def extract(self, pdf_path):
        output_folder = '/home/bal/Desktop/pdf-extraction/detectron'
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        pages = pdf2image.convert_from_path(pdf_path)
        results = []

        for page_num, page in enumerate(pages):
            img = np.asarray(page)
            img = img.copy()  # Ensure the image is not read-only
            layout_result = self.model.detect(img)
            page_results = []
            for block in layout_result:
                left, top, right, bottom = map(int, block.coordinates)
                cropped_img = img[top:bottom, left:right]

                extracted_text = pytesseract.image_to_string(cropped_img, lang='eng').strip()
                page_results.append({
                    'page': page_num + 1,
                    'left': left,
                    'top': top,
                    'right': right,
                    'bottom': bottom,
                    'label': block.type,
                })

                # Draw rectangle around the detected area
                cv2.rectangle(img, (left, top), (right, bottom), (0, 255, 0), 2)
                cv2.putText(img, block.type, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            results.append(page_results)
            output_path = os.path.join(output_folder, f'page_{page_num + 1}.jpg')
            cv2.imwrite(output_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            print(f"Saved annotated image: {output_path}")

        return results
