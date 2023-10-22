import json
import os
from model.ressource import Ressource, Category

class RessourcesManager:
    
    def __init__(self):
        self.data = []  # list of Category
        self.__init_ressources()
        
    
    def __init_ressources(self):
        file_dir = os.path.dirname(os.path.abspath(__file__))
        file = os.path.join(file_dir, "data", "ressources.json")
        
        # Load the JSON data from a file (or you can load it from a string)
        with open(file, "r", encoding="utf-8") as json_file:
            json_data = json.load(json_file)

        for category_data in json_data.get('data', []):
            category = Category(
                name=category_data.get('name', ''),
                ressources=[
                    Ressource(
                        name=res_data.get('name', ''),
                        pretty_name=res_data.get('pretty_name', ''),
                        enabled=False,
                        id=res_data.get('id', -1)
                    )
                    for res_data in category_data.get('ressources', [])
                ]
            )
            self.data.append(category)
    
    def get_ressources(self, category: str) -> [Ressource]:
        for cat in self.data:
            if cat.name == category:
                return cat.ressources
        return None
    
    def change_global_status(self, active_cat: str, status: bool) -> None:
        for cat in self.data:
            if cat.name == active_cat:
                for res in cat.ressources:
                    res.enabled = status