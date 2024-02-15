from abc import ABC, abstractmethod

class Observable:
    """The observable interface"""

    @property
    def observers(self):
        """Oberservers looking for the observable obect"""
        try:
            return self.__observers
        except AttributeError:
            # here self.__observers has not been created yet
            self.__observers: list[_IObserver] = []
            return self.__observers
    
    def _add_observer(self, observer) -> None:
        """Add the observer."""

        if not isinstance(observer, _IObserver):
            from Display import myPrintError
            myPrintError(f'observer must be an {_IObserver.__name__}')
            return
        
        if observer not in self.observers:
            self.__observers.append(observer)

    def _remove_observer(self, observer) -> None:
        """Remove the observer."""

        if observer in self.observers:
            self.__observers.remove(observer)

    def _notify(self, event: str) -> None:
        """Notifies the observers"""

        [observer.update(self, event) for observer in self.observers]

class _IObserver(ABC):
    """The observer interface"""
    @abstractmethod
    def update(self, observable: Observable, event: str) -> None:
        """Receive an update from an observable object."""        
        pass