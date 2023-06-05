import customtkinter as ctk
from PIL import Image, ImageTk
import tkinter as tk
import joblib
import webbrowser
import src.convert_url_to_csv as to_csv

import pandas as pd
import src.constant as C


class MainApplication(tk.Tk):
    def __init__(self, *args, **kwargs):
        tk.Tk.__init__(self, *args, **kwargs)
        self.geometry("900x506")
        self.resizable(0, 0)
        self.title('PROJET CLUSTERING')
        
        self.bg_images = {
            'start': ImageTk.PhotoImage(Image.open("image/VÉRIFIER VOTRE URL.png")),
            'result1': ImageTk.PhotoImage(Image.open("image/URL VÉRIFIÉ1.png")),
            'result31': ImageTk.PhotoImage(Image.open("image/URL VÉRIFIÉ_malware.png")),
            'result32': ImageTk.PhotoImage(Image.open("image/URL VÉRIFIÉ_phishing.png")),
            'result33': ImageTk.PhotoImage(Image.open("image/URL VÉRIFIÉ_spam.png")),
            'result34': ImageTk.PhotoImage(Image.open("image/URL VÉRIFIÉ_defacement.png")),
            'result3': ImageTk.PhotoImage(Image.open("image/URL VÉRIFIÉ3.png")),
            'detail': {
                'benign': ImageTk.PhotoImage(Image.open("image/DÉTAILS_benign.png")),
                'malware': ImageTk.PhotoImage(Image.open("image/DÉTAILS_malware.png")),
                'phishing': ImageTk.PhotoImage(Image.open("image/DÉTAILS_phishing.png")),
                'spam': ImageTk.PhotoImage(Image.open("image/DÉTAILS_spam.png")),
                'defacement': ImageTk.PhotoImage(Image.open("image/DÉTAILS_defacement.png"))
            }
        }

        # The result type is benign by default
        self.result_type = 'benign'

        # container pour les frames
        container = tk.Frame(self)
        container.pack(side="top", fill="both", expand=True)
        container.grid_rowconfigure(0, weight=1)
        container.grid_columnconfigure(0, weight=1)

        # Initialize empty dictionary to store frames
        self.frames = {}

        for F in (StartPage, ResultPage1, ResultPage3, TreatmentDetailsPage):
            frame = F(container, self, self.bg_images)
            self.frames[F] = frame
            frame.grid(row=0, column=0, sticky="nsew")

        self.show_frame(StartPage)

    def show_frame(self, page_name):
        frame = self.frames[page_name]
        frame.tkraise()


class StartPage(tk.Frame):
    def __init__(self, parent, controller, bg_images):
        tk.Frame.__init__(self, parent)
        self.controller = controller
        self.fond = ctk.CTkLabel(master=self, image=bg_images['start'])
        self.fond.place(x=0, y=0)
        self.url1 = ""

        self.url1produit = ctk.CTkEntry(master=self, width=400, bg_color=("#000000", "#000000"), fg_color=("#FFFFFF", "#FFFFFF"))
        self.url1produit.insert(0, "ENTRER UN URL")
        self.url1produit.bind("<FocusIn>", self.clear_placeholder)
        self.url1produit.place(x=250, y=230)

        self.valider = ctk.CTkButton(master=self, text='VALIDER', command=self.chargement, bg_color=("#000000", "#000000"), fg_color=("#000000", "#000000"))
        self.valider.place(x=380, y=300)

    def chargement(self):
        clf=joblib.load('../../result/rf/all.pkl')
        url1 = self.url1produit.get()

        data=[]
        
        data.append(to_csv.url_to_dico(url1))
        df = pd.DataFrame(data, columns=C.COLUMNS)
        df.to_csv('../../result/prediction/test.csv', index=False)

      
        url1 = clf.predict(df)
        print(url1)
        url1 = str(url1[0])

        if url1 == "0":
            result = "benign"
        elif url1 == "1":
            resutl = "spam"
        elif url1 == "2":
            result = "malware"
        elif url1 == "3":
            result = "phishing"
        elif url1 == "4":
            result = "defacement"
        print(result)
        self.url1 = url1

        if result == "benign":
            self.controller.show_frame(ResultPage1)
        else:
            self.result_type = result
            # Pass url1 to ResultPage3
            self.controller.frames[ResultPage3].update_image(result)
            self.controller.show_frame(ResultPage3)

    def clear_placeholder(self, event):
        if self.url1produit.get() == 'ENTRER UN URL':
            self.url1produit.delete(0, 'end')


class ResultPage1(tk.Frame):
    def __init__(self, parent, controller, bg_images):
        tk.Frame.__init__(self, parent)
        self.controller = controller
        self.start_page = controller.frames[StartPage]
        self.fond = ctk.CTkLabel(master=self, image=bg_images['result1'])
        self.fond.place(x=0, y=0)

        ctk.CTkButton(self, text="ACCÉDER AU LIEN", command=self.open_link, bg_color=("#000000", "#000000"), fg_color=("#000000", "#000000")).place(x=175, y=250)
        ctk.CTkButton(self, text="RETOUR AU MENU", command=lambda: controller.show_frame(StartPage), bg_color=("#000000", "#000000"), fg_color=("#000000", "#000000")).place(x=575, y=250)

        ctk.CTkButton(self, text="DÉTAILS DU TRAITEMENT", command=lambda: controller.show_frame(TreatmentDetailsPage), bg_color=("#000000", "#000000"), fg_color=("#000000", "#000000")).place(x=375, y=350)

    def open_link(self):
        webbrowser.open(self.start_page.url1)


class ResultPage3(tk.Frame):
    def __init__(self, parent, controller, bg_images):
        tk.Frame.__init__(self, parent)
        self.controller = controller
        self.start_page = controller.frames[StartPage]
        self.bg_images = bg_images
        self.fond = ctk.CTkLabel(master=self, image=bg_images['result3'])
        self.fond.place(x=0, y=0)

        ctk.CTkButton(self, text="RETOUR AU MENU", command=lambda: controller.show_frame(StartPage), bg_color=("#000000", "#000000"), fg_color=("#000000", "#000000")).place(x=385, y=300)
        ctk.CTkButton(self, text="DÉTAILS DU TRAITEMENT", command=lambda: controller.show_frame(TreatmentDetailsPage), bg_color=("#000000", "#000000"), fg_color=("#000000", "#000000")).place(x=375, y=350)

    def update_image(self, url1):

        if url1 == "malware":
            self.fond.configure(image=self.bg_images['result31'])
        elif url1 == "phishing":
            self.fond.configure(image=self.bg_images['result32'])
        elif url1 == "spam":
            self.fond.configure(image=self.bg_images['result33'])
        elif url1 == "defacement":
            self.fond.configure(image=self.bg_images['result34'])

        self.fond.place(x=0, y=0)


class TreatmentDetailsPage(tk.Frame):
    def __init__(self, parent, controller, bg_images):
        tk.Frame.__init__(self, parent)
        self.controller = controller
        self.bg_images = bg_images
        self.result_type = controller.result_type

        self.fond = ctk.CTkLabel(master=self, image=self.bg_images['detail'][self.result_type])
        self.fond.place(x=0, y=0)

        ctk.CTkButton(self, text="Retour", command=lambda: controller.show_frame(ResultPage3), bg_color=("#000000", "#000000"), fg_color=("#000000", "#000000")).place(x=380, y=400)
        ctk.CTkButton(self, text="PAGE SUIVANTE", command=self.next_page, bg_color=("#000000", "#000000"), fg_color=("#000000", "#000000")).place(x=680, y=400)
        ctk.CTkButton(self, text="PAGE PRÉCÉDENTE", command=self.prev_page, bg_color=("#000000", "#000000"), fg_color=("#000000", "#000000")).place(x=100, y=400)

    def next_page(self):
        # Implement your function to go to the next page here
        pass

    def prev_page(self):
        # Implement your function to go to the previous page here
        pass
if __name__ == "__main__":
    app = MainApplication()
    app.mainloop()