class Ressource:
    def __init__(self, name: str, pretty_name: str, enabled: bool, id: int):
        self.name = name
        self.pretty_name = pretty_name
        self.enabled = enabled
        self.id = id
        
class Category:
    def __init__(self, name: str, ressources: [Ressource]):
        self.name = name
        self.ressources = ressources