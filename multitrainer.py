import threading
import model

class MultiTrainer:
    def __init__(self, mdl_list: list[model.Model]) -> None:
        self.mdls = mdl_list
        self.results = {}
        self.threads = []
        return
    
    def __trainmodel__(self, mdl: model.Model) -> None:
        """ Wrapper for model.Model.train() """

        # Send return value to self.results
        self.results[mdl.id] = mdl.train()
        return

    def train_models(self):
        # Start new thread for each model
        for mdl in self.mdls:
            thread = threading.Thread(
                target=self.__trainmodel__,
                args=(mdl,)
            )
            self.threads.append(thread)
            thread.start()

        for thread in self.threads:
            thread.join()
        
        return self.results