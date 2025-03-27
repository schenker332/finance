import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
from segment_anything import sam_model_registry, SamPredictor

bboxes = []

# Configuration of the image path
image_dir = '/Users/niclas/VS Code/pics'
image_names = [f for f in os.listdir(image_dir) if f.endswith('.png')]

# Configuration of the SAM model
sam_dir = '/Users/niclas/VS Code'
sam_checkpoint = os.path.join(sam_dir, 'sam_vit_h_4b8939.pth')
model_type = 'vit_h'
device = 'mps'
sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)
predictor = SamPredictor(sam)

def get_points_from_image(image_name):
    image_path = os.path.join(image_dir, image_name)
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    points = []

    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    ax.imshow(image)
    ax.set_title("Raw Image", fontsize=20)
    
    def onclick(event):
        if event.inaxes is not None:
            if len(points) < 3:
                points.append((int(event.xdata), int(event.ydata)))
                ax.plot(event.xdata, event.ydata, 'ro')
                fig.canvas.draw()
            if len(points) == 3:
                plt.close(fig)

    fig.canvas.mpl_connect('button_press_event', onclick)
    plt.show()

    return np.array(points), np.array([1, 1, 1]), image

def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    if pos_points.size > 0:
        ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    if neg_points.size > 0:
        ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   
    
def create_mask(image, input_point, input_label):
    predictor.set_image(image)

    masks, scores, logits = predictor.predict(
        point_coords=input_point,
        point_labels=input_label,
        multimask_output=True,
    )

    top_mask = masks[np.argmax(scores), :, :]

    return top_mask

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)   

def mask_to_bbox(mask):
    y_indices, x_indices = np.where(mask)
    x_min, x_max = x_indices.min(), x_indices.max()
    y_min, y_max = y_indices.min(), y_indices.max()
    width = x_max - x_min
    height = y_max - y_min
    return [int(x_min), int(y_min), int(width), int(height)]

def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2], box[3]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='black', facecolor=(0,0,0,0), lw=4))

for image_name in image_names:
    print(image_name)
    # Input three representing object points
    input_points, input_label, image = get_points_from_image(image_name)

    # Create a plot to show only the points
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    ax.imshow(image)
    show_points(input_points, input_label, ax)
    ax.set_title("Points", fontsize=20)
    plt.show()

    
    # Create a mask using the points
    mask = create_mask(image, input_points, input_label)

    # Create a plot to show only the mask
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    ax.imshow(image)
    show_mask(mask, ax)
    ax.set_title("Mask", fontsize=20)
    plt.show()

    # Create a bounding box using the object mask
    bbox = mask_to_bbox(mask)
    bboxes.append(bbox)
    print(f"Bbox: {bbox}")

    # Creat a plot using the bbox
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    ax.imshow(image)
    show_box(bbox, ax)
    ax.set_title('Bounding Box', fontsize=20)
    plt.show()
    break

print(image_names)
print(bboxes)