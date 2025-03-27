# # Horizontale Linien
# horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (60, 1))
# detected_horizontal = cv2.morphologyEx(edges, cv2.MORPH_OPEN, horizontal_kernel, iterations=1)
# contours_h, _ = cv2.findContours(detected_horizontal, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)



# for i, contour in enumerate(contours_h):
#     x, y, w, h = cv2.boundingRect(contour)
#     width.append(w)
#     y_coords.append(y)
#     # print(f"Linie {i+1}: x={x}, y={y}, Breite={w}px")




# Vertikale Linien
# height = []
# width = []

# x_coords = []
# y_coords = []

# vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 50))
# detected_vertical = cv2.morphologyEx(edges, cv2.MORPH_OPEN, vertical_kernel, iterations=1)
# contours_v, _ = cv2.findContours(detected_vertical, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)


# for i, contour in enumerate(contours_v):
#     x, y, w, h = cv2.boundingRect(contour)
#     height.append(h)
#     x_coords.append(x)
#     # print(f"Linie {i+1}: x={x}, y={y},  Höhe={h}px")

