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
        self.all_athletes_seen: Set[str] = set()
        self.events: List[EventRecord] = []
        
        self.frames_processed = 0
        self.frames_with_detection = 0
        
        self.pixel_buffer = 40.0 
        self.swap_cooldown = 30
        
        # New Pairwise Tracking State
        self.confirmed_order: Dict[Tuple[str, str], int] = {} 
        self.pending_swaps: Dict[Tuple[str, str], Dict] = {}

    def reset_state(self, frame_number: int, frame_name: str, trigger: str = "camera_cut"):
        self._record_event(frame_number, frame_name, [], trigger)
        # Clear out state on camera cuts so we don't accidentally bridge cuts
        self.confirmed_order.clear()
        self.pending_swaps.clear()

    def detect_swaps(self, current_positions: Dict[str, float], frame_name: str, frame_number: int) -> List[Tuple[str, str]]:
        confirmed_swaps = []
        common_athletes = sorted(list(current_positions.keys()))

        # Check every pair of athletes currently on stage
        for i in range(len(common_athletes)):
            for j in range(i + 1, len(common_athletes)):
                a, b = common_athletes[i], common_athletes[j]
                
                # Always order the pair alphabetically so the dictionary key is consistent
                pair = tuple(sorted([a, b]))
                p1, p2 = pair
                
                curr_diff = current_positions[p1] - current_positions[p2]
                
                # 1 if p1 is right of p2, -1 if left. 0 if dead center (ignore)
                if curr_diff > self.pixel_buffer:
                    current_state = 1
                elif curr_diff < -self.pixel_buffer:
                    current_state = -1
                else:
                    current_state = 0 
                
                if current_state == 0:
                    continue
                    
                # First time seeing this pair? Log their order and move on
                if pair not in self.confirmed_order:
                    self.confirmed_order[pair] = current_state
                    self.pending_swaps[pair] = {'state': current_state, 'frames': 0}
                    continue
                
                if current_state != self.confirmed_order[pair]:
                    if pair in self.pending_swaps and self.pending_swaps[pair]['state'] == current_state:
                        self.pending_swaps[pair]['frames'] += 1
                    else:
                        # Start tracking a new pending swap
                        self.pending_swaps[pair] = {'state': current_state, 'frames': 1}
                        
                    if self.pending_swaps[pair]['frames'] >= self.swap_cooldown:
                        self.confirmed_order[pair] = current_state
                        confirmed_swaps.append((p1, p2))
                        self.pending_swaps[pair] = {'state': current_state, 'frames': 0}
                        
                        print("=" * 50)
                        print(f" SWAP CONFIRMED: Frame {frame_number}")
                        print(f"   {p1} <-> {p2}")
                        print("=" * 50)
                else:
                    self.pending_swaps[pair] = {'state': self.confirmed_order[pair], 'frames': 0}

        return confirmed_swaps

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
        
        # 1. Update our memory with anyone currently visible
        for athlete, pos in current_positions.items():
            self.last_known_positions[athlete] = pos
            
        # 2. If they are missing, force them back into the lineup
        for athlete in list(self.last_known_positions.keys()):
            if athlete not in current_positions:
                current_positions[athlete] = self.last_known_positions[athlete]

        # 3. Track detection stats
        if current_positions:
            self.frames_with_detection += 1
            
        # 4. Log newly seen athletes
        for athlete in current_positions.keys():
            if athlete not in self.all_athletes_seen:
                self.all_athletes_seen.add(athlete)
                print(f"  [New Athlete]: {athlete}")

        # 5. Calculate the order and detect any crossovers!
        ordered_athletes = self._get_ordered_athletes(current_positions)
        swaps = self.detect_swaps(current_positions, frame_name, frame_number)

        # 6. Log the event if a swap happened
        if swaps:
            self._record_event(frame_number, frame_name, ordered_athletes, "position_changed", swaps)

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