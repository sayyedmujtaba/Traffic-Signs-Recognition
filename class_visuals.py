import matplotlib.pyplot as plt  # for plotting the chart

class_counts = {  # put your values here (from earlier count script output)
# obtained from running the count_class.py script
# result is stored in Train/result.md
    0: 210, 1: 2220, 2: 2250, 3: 1410, 4: 1980, 5: 1860, 6: 420, 7: 1440,
    8: 1410, 9: 1470, 10: 2010, 11: 1320, 12: 2100, 13: 2160, 14: 780,
    15: 630, 16: 420, 17: 1110, 18: 1200, 19: 210, 20: 360, 21: 330,
    22: 390, 23: 510, 24: 270, 25: 1500, 26: 600, 27: 240, 28: 540,
    29: 270, 30: 450, 31: 780, 32: 240, 33: 689, 34: 420, 35: 1200,
    36: 390, 37: 210, 38: 2070, 39: 300, 40: 360, 41: 240, 42: 240
}

sorted_items = sorted(class_counts.items(), key=lambda x: x[1])  # sort by count, smallest to largest
classes = [str(item[0]) for item in sorted_items]  # extract class IDs
counts = [item[1] for item in sorted_items]  # extract counts

plt.figure(figsize=(14, 6))  # set figure size for readability
plt.bar(classes, counts, color='steelblue')  # draw bar chart
plt.xlabel("Class ID")  # label x-axis
plt.ylabel("Number of Images")  # label y-axis
plt.title("Class Distribution in Train Set (sorted)")  # chart title
plt.xticks(rotation=90)  # rotate x labels to avoid overlap
plt.tight_layout()  # fix spacing
plt.savefig("class_distribution.png")  # save chart as image file
plt.show()  # display chart