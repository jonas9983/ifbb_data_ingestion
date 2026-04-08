from typing import Dict, List, Set, Tuple, Optional
from dataclasses import dataclass, asdict
from datetime import datetime

@dataclass
class EventRecord:
    frame_number: int
    frame_name: str
    timestamp: str
    athletes: List[str] 
    trigger: str
    athletes_swapped: Optional[List[Tuple[str, str]]] = None
    
    def to_dict(self):
        data = asdict(self)
        if self.athletes_swapped:
            data['athletes_swapped'] = [{'athlete_1': a, 'athlete_2': b} for a, b in self.athletes_swapped]
        return data

class TrackingEventLogger:
    def __init__(self):
        self.last_frame_positions: Dict[str, float] = {}
        self.all_athletes_seen: Set[str] = set()
        self.events: List[EventRecord] = []
        
        self.frames_processed = 0
        self.frames_with_detection = 0
        
        self.last_swap_frame = -1
        self.swap_cooldown = 10 
        
        self.pixel_buffer = 40.0 

    def reset_state(self, frame_number: int, frame_name: str, trigger: str = "camera_cut"):
        self._record_event(frame_number, frame_name, [], trigger)
        self.last_frame_positions.clear()
        self.last_swap_frame = -1

    def detect_swaps(self, current_positions: Dict[str, float], frame_name: str, frame_number: int) -> List[Tuple[str, str]]:
        swaps = []
        common_athletes = sorted(list(set(self.last_frame_positions.keys()) & set(current_positions.keys())))
        
        if len(common_athletes) < 2:
            return swaps

        for i in range(len(common_athletes)):
            for j in range(i + 1, len(common_athletes)):
                a, b = common_athletes[i], common_athletes[j]
                
                prev_diff = self.last_frame_positions[a] - self.last_frame_positions[b]
                curr_diff = current_positions[a] - current_positions[b]
                
                # Check for crossing with a buffer to prevent jitter spam
                swapped = False
                if prev_diff < -self.pixel_buffer and curr_diff > self.pixel_buffer:
                    swapped = True
                elif prev_diff > self.pixel_buffer and curr_diff < -self.pixel_buffer:
                    swapped = True

                if swapped:
                    swaps.append((a, b))
                    
                    if frame_number - self.last_swap_frame > self.swap_cooldown:
                        print("=" * 50)
                        print(f"🔄 SWAP CONFIRMED: {frame_name}")
                        print(f"   {a} <-> {b}")
                        print("=" * 50)
                        self.last_swap_frame = frame_number

        return swaps

    def _get_ordered_athletes(self, positions: Dict[str, float]) -> List[str]:
        return [name for name, _ in sorted(positions.items(), key=lambda x: x[1])]

    def _record_event(self, frame_number: int, frame_name: str, athletes: List[str], trigger: str, swaps: Optional[List] = None):
        self.events.append(EventRecord(
            frame_number=frame_number, frame_name=frame_name, 
            timestamp=datetime.now().isoformat(), athletes=athletes, 
            trigger=trigger, athletes_swapped=swaps
        ))

    def update_state(self, current_positions: Dict[str, float], frame_name: str, frame_number: int) -> Dict:
        self.frames_processed += 1
        if current_positions:
            self.frames_with_detection += 1
            
        for athlete in current_positions.keys():
            if athlete not in self.all_athletes_seen:
                self.all_athletes_seen.add(athlete)
                print(f" [New Athlete]: {athlete}")

        ordered_athletes = self._get_ordered_athletes(current_positions)
        swaps = self.detect_swaps(current_positions, frame_name, frame_number)

        if swaps:
            self._record_event(frame_number, frame_name, ordered_athletes, "position_changed", swaps)
            self.last_frame_positions = current_positions
        elif current_positions:
            prev_lineup = self._get_ordered_athletes(self.last_frame_positions)
            if ordered_athletes != prev_lineup and len(ordered_athletes) >= len(prev_lineup):
                self._record_event(frame_number, frame_name, ordered_athletes, "lineup_changed")
            self.last_frame_positions = current_positions

        return {'swaps': swaps, 'ordered_athletes': ordered_athletes}

    def get_statistics(self) -> Dict:
        return {
            'frames_processed': self.frames_processed,
            'frames_with_detection': self.frames_with_detection,
            'total_athletes': len(self.all_athletes_seen),
            'athletes': sorted(list(self.all_athletes_seen)),
            'total_events': len(self.events)
        }

    def print_summary(self):
        stats = self.get_statistics()
        print("\n" + "=" * 50)
        print(" TRACKING SUMMARY")
        print("=" * 50)
        print(f"Processed: {stats['frames_processed']} | Events: {stats['total_events']}")
        print(f"Unique Athletes ({stats['total_athletes']}): {', '.join(stats['athletes'])}")
        print("=" * 50)