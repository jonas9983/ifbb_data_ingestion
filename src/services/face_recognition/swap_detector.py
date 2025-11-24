"""
swap_detector.py

Handles the logic for detecting when athletes swap positions on stage.
"""

from typing import Dict, List, Set, Tuple, Optional
from dataclasses import dataclass, asdict
from datetime import datetime


@dataclass
class EventRecord:
    """Represents a tracking event."""
    frame_number: int
    frame_name: str
    timestamp: str
    athletes: List[str]   # Ordered left to right
    trigger: str
    athletes_swapped: Optional[List[Tuple[str, str]]] = None
    
    def to_dict(self):
        """Convert to dictionary for JSON serialization."""
        data = asdict(self)
        # Convert list of tuples to more readable format for JSON
        if self.athletes_swapped:
            data['athletes_swapped'] = [
                {'athlete_1': a, 'athlete_2': b} 
                for a, b in self.athletes_swapped
            ]
        return data


class SwapDetector:
    """
    Detects when athletes swap positions based on their x-coordinates.
    Maintains state across frames and records swap events.
    """
    
    def __init__(self):
        """Initialize the swap detector."""
        # State tracking: Stores the most recent, stable positions (Name: Center_X)
        self.last_frame_positions: Dict[str, float] = {}
        self.all_athletes_seen: Set[str] = set()
        
        # Event recording
        self.events: List[EventRecord] = []
        
        # Statistics
        self.frames_with_detection: int = 0
        self.frames_processed: int = 0
        
    def reset_state(
        self, 
        frame_number: int, 
        frame_name: str, 
        trigger: str = "state_reset"
    ):
        """
        Clears the last known stable positions and records an event.
        Used primarily after a camera cut is detected.
        """
        
        # Record the reset event
        self.record_event(
            frame_number=frame_number,
            frame_name=frame_name,
            athletes=[], # No lineup recorded for a reset event
            trigger=trigger,
            swaps=None
        )

        # CRUCIAL STEP: Clear historical state
        self.last_frame_positions = {}
        # We DO NOT clear self.all_athletes_seen as they are still known.
    
    def update_athletes_seen(self, athletes: List[str]):
        """Track new athletes that appear in the frame."""
        for athlete in athletes:
            if athlete not in self.all_athletes_seen:
                self.all_athletes_seen.add(athlete)
                print(f"[New Athlete Detected]: {athlete}")
    
    def detect_swaps(
        self, 
        current_positions: Dict[str, float], 
        frame_name: str
    ) -> List[Tuple[str, str]]:
        """
        Detect if any athletes swapped positions since last frame.
        
        Returns:
            List of tuples (athlete_a, athlete_b) that swapped
        """
        swaps = []
        
        # Need at least 2 common athletes in both frames for a meaningful comparison
        if len(self.last_frame_positions) < 2 or len(current_positions) < 2:
            return swaps
        
        # Find athletes present in BOTH historical state and current state
        common_athletes = set(self.last_frame_positions.keys()) & set(current_positions.keys())
        
        if len(common_athletes) < 2:
            return swaps
        
        # Get lineup names for clear logging
        prev_lineup_names = self.get_ordered_athletes(self.last_frame_positions)
        current_lineup_names = self.get_ordered_athletes(current_positions)
        
        # Check all pairs for position swaps among common athletes
        common_list = sorted(list(common_athletes))
        
        for i in range(len(common_list)):
            for j in range(i + 1, len(common_list)):
                athlete_a = common_list[i]
                athlete_b = common_list[j]
                
                # Previous positions from the last stable frame
                prev_a = self.last_frame_positions[athlete_a]
                prev_b = self.last_frame_positions[athlete_b]
                
                # Current positions
                curr_a = current_positions[athlete_a]
                curr_b = current_positions[athlete_b]
                
                # Check if relative order changed (A was left of B, now A is right of B, etc.)
                prev_order = "A_left_of_B" if prev_a < prev_b else "B_left_of_A"
                curr_order = "A_left_of_B" if curr_a < curr_b else "B_left_of_A"
                
                if prev_order != curr_order:
                    # Log the swap detection with clear text instead of arrows
                    prev_relative = f"{athlete_a} was on the {'LEFT' if prev_a < prev_b else 'RIGHT'} of {athlete_b}"
                    curr_relative = f"{athlete_a} is now on the {'LEFT' if curr_a < curr_b else 'RIGHT'} of {athlete_b}"
                    
                    print("=" * 60)
                    print(f"🎉 POSITION SWAP DETECTED in: {frame_name}")
                    print(f"   Between: {athlete_a} and {athlete_b}")
                    print(f"   ---")
                    print(f"   PREVIOUS FULL LINEUP: {' | '.join(prev_lineup_names)}") 
                    print(f"   CURRENT FULL LINEUP:  {' | '.join(current_lineup_names)}") 
                    print(f"   ---")
                    print(f"   Relative Position Change:")
                    print(f"     Previous: {prev_relative}")
                    print(f"     Current:  {curr_relative}")
                    print("=" * 60)
                    swaps.append((athlete_a, athlete_b))
        
        return swaps
    
    def get_ordered_athletes(self, positions: Dict[str, float]) -> List[str]:
        """Get list of athlete names ordered left to right."""
        sorted_athletes = sorted(positions.items(), key=lambda x: x[1])
        return [name for name, _ in sorted_athletes]
    
    def record_event(
        self, 
        frame_number: int,
        frame_name: str,
        athletes: List[str],
        trigger: str,
        swaps: Optional[List[Tuple[str, str]]] = None
    ):
        """Record a tracking event."""
        event = EventRecord(
            frame_number=frame_number,
            frame_name=frame_name,
            timestamp=datetime.now().isoformat(),
            athletes=athletes,
            trigger=trigger,
            athletes_swapped=swaps if swaps else None
        )
        self.events.append(event)
    
    def update_state(
        self, 
        current_positions: Dict[str, float],
        frame_name: str,
        frame_number: int
    ) -> Dict:
        """
        Update state and detect changes.
        """
        self.frames_processed += 1
        
        if len(current_positions) > 0:
            self.frames_with_detection += 1
        
        self.update_athletes_seen(list(current_positions.keys()))
        ordered_athletes = self.get_ordered_athletes(current_positions)
        
        # 1. Detect Swaps by comparing current vs last stable state
        swaps = self.detect_swaps(current_positions, frame_name)
        
        # 2. Record Events
        if swaps:
            # Position change event (Actual swap detected)
            self.record_event(frame_number, frame_name, ordered_athletes, "position_changed", swaps)
        elif len(ordered_athletes) > 0:
            prev_lineup = self.get_current_lineup_names()
            
            # Check for lineup changes (athletes appeared/disappeared)
            if ordered_athletes != prev_lineup and len(ordered_athletes) >= len(prev_lineup):
                self.record_event(frame_number, frame_name, ordered_athletes, "lineup_changed")
            
            # Log an event for every detected frame if the lineup is stable
            elif ordered_athletes == prev_lineup:
                self.record_event(frame_number, frame_name, ordered_athletes, "stable_lineup")
        
        # 3. Update State (Intelligent Update)
        # We only update if the current state is stable or growing.
        if len(current_positions) > 0:
            if len(current_positions) >= len(self.last_frame_positions):
                self.last_frame_positions = current_positions
        
        return {
            'swaps': swaps,
            'ordered_athletes': ordered_athletes
        }
    
    def get_current_lineup(self) -> List[Tuple[int, str]]:
        """Get the current lineup from left to right with positions."""
        sorted_athletes = sorted(
            self.last_frame_positions.items(), 
            key=lambda x: x[1]
        )
        return [(idx + 1, name) for idx, (name, _) in enumerate(sorted_athletes)]
    
    def get_current_lineup_names(self) -> List[str]:
        """Get the current lineup as a list of names (left to right)."""
        sorted_athletes = sorted(
            self.last_frame_positions.items(), 
            key=lambda x: x[1]
        )
        return [name for name, _ in sorted_athletes]
    
    def get_statistics(self) -> Dict:
        """Get tracking statistics."""
        return {
            'frames_processed': self.frames_processed,
            'frames_with_detection': self.frames_with_detection,
            'total_athletes': len(self.all_athletes_seen),
            'athletes': sorted(list(self.all_athletes_seen)),
            'total_events': len(self.events)
        }
    
    def print_summary(self):
        """Print a summary of tracking results."""
        stats = self.get_statistics()
        
        print("\n" + "=" * 60)
        print("📊 TRACKING SUMMARY")
        print("=" * 60)
        print(f"Frames processed: {stats['frames_processed']}")
        print(f"Frames with recognized faces: {stats['frames_with_detection']}")
        
        if stats['frames_processed'] > 0:
            detection_rate = stats['frames_with_detection'] / stats['frames_processed'] * 100
            print(f"Detection rate: {detection_rate:.1f}%")
        
        print(f"\nTotal unique athletes detected: {stats['total_athletes']}")
        print("\nAthletes identified:")
        for idx, athlete in enumerate(stats['athletes'], 1):
            print(f"  {idx}. {athlete}")
        
        if len(self.last_frame_positions) > 0:
            print(f"\nFinal lineup (left to right):")
            for position, name in self.get_current_lineup():
                print(f"  Position {position}: {name}")
        
        print(f"\nTotal events recorded: {stats['total_events']}")
        print("=" * 60)