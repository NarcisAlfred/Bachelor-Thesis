import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# directory of the excel file
excel = r"C:\Users\Alfred\Desktop\Importante\Licenta\database\testing\Data.xlsx"

# output directory
output = r"C:\Users\Alfred\Desktop\Importante\Licenta\Visual Studio\CNN\output"

# Initialize empty lists
actual_edv = []
predicted_edv = []

actual_esv = []
predicted_esv = []

actual_ef = []
predicted_ef = []


# read the excel
df = pd.read_excel(excel)

# iterate through each row
for index, row in df.iterrows():
    actual_edv.append(row["EDV"])
    predicted_edv.append(row["Predicted EDV"])

    actual_esv.append(row["ESV"])
    predicted_esv.append(row["Predicted ESV"])

    actual_ef.append(row["EF"])
    predicted_ef.append(row["Predicted EF"])

# Save plots
# EDV
plt.scatter(predicted_edv, actual_edv)
plt.xlabel('Predicted EDV (mL)')
plt.ylabel('Actual EDV (mL)')
plt.title('End dyastolic volume')
plt.grid(True)
plt.savefig(output+"\EDV_plot.png")
plt.close()

# ESV
plt.scatter(predicted_esv, actual_esv)
plt.xlabel('Predicted ESV (mL)')
plt.ylabel('Actual ESV (mL)')
plt.title('End sistolic volume')
plt.grid(True)
plt.savefig(output+"\ESV_plot.png")
plt.close()

# EDV
plt.scatter(predicted_ef, actual_ef)
plt.xlabel('Predicted EF (%)')
plt.ylabel('Actual EF (%)')
plt.title('Ejection fraction volume')
plt.grid(True)
plt.savefig(output+"\EF_plot.png")