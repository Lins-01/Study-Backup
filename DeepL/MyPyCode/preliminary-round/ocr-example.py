from paddleocr import PaddleOCR, draw_ocr

# Paddleocr目前支持的多语言语种可以通过修改lang参数进行切换
# 例如`ch`, `en`, `fr`, `german`, `korean`, `japan`
# ocr = PaddleOCR(use_angle_cls=True, lang="ch")  # need to run only once to download and load model into memory
ocr = PaddleOCR(use_angle_cls=True, lang="ch", use_gpu=False, use_mp=True, total_process_num=6, show_log=False)
img_path = r'E:\Document\CodeSpace\Data_set\Paddle2023IKCEST\queries_dataset_merge/val\img_html_news\1\1275.jpg'
# img_path = str(img_path)
# img_path = os.path.abspath(img_path)
print(type(img_path))
print(img_path)
result = ocr.ocr(img_path, cls=False)
for idx in range(len(result)):
    res = result[idx]
    for line in res:
        print(line)

print('done')
# 显示结果
from PIL import Image
result = result[0]
image = Image.open(img_path).convert('RGB')
boxes = [line[0] for line in result]
txts = [line[1][0] for line in result]
scores = [line[1][1] for line in result]
im_show = draw_ocr(image, boxes, txts, scores, font_path='Deng.ttf')# 要出图，随便下载个字体
im_show = Image.fromarray(im_show)
im_show.save('result.jpg')
