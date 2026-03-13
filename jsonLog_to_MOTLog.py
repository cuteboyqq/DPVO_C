import json
import logging
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(message)s')

'''
Taken from: https://motchallenge.net/instructions/

File Format

Please submit your results as a single .zip file.
The results for each sequence must be stored in a separate .
txt file in the archive's root folder. 
The file name must be exactly like the sequence name (case sensitive).

The file format should be the same as the ground truth file, 
which is a CSV text-file containing one object instance per line. Each line must contain 10 values:

<frame>, <id>, <bb_left>, <bb_top>, <bb_width>, <bb_height>, <conf>, <x>, <y>, <z>
The conf value contains the detection confidence in the det.txt files. For the ground truth, 
it acts as a flag whether the entry is to be considered. 
A value of 0 means that this particular instance is ignored in the evaluation,
while any other value can be used to mark it as active.
For submitted results, all lines in the .txt file are considered. 
The world coordinates x,y,z are ignored for the 2D challenge and can be filled with -1.
Similarly, the bounding boxes are ignored for the 3D challenge. However, each line is still required to contain 10 values.

All frame numbers, target IDs and bounding boxes are 1-based. Here is an example:

Tracking with bounding boxes
(MOT15, MOT16, MOT17, MOT20)
  1, 3, 794.27, 247.59, 71.245, 174.88, -1, -1, -1, -1
  1, 6, 1648.1, 119.61, 66.504, 163.24, -1, -1, -1, -1
  1, 8, 875.49, 399.98, 95.303, 233.93, -1, -1, -1, -1
  ...
'''

MOT15_RESOLUTION_DICT = {
    "ADL-Rundle-6": (1920, 1080),
    "ADL-Rundle-8": (1920, 1080),
    "ETH-Bahnhof": (640, 480),
    "ETH-Pedcross2": (640, 480),
    "ETH-Sunnyday": (640, 480),
    "KITTI-13": (1242, 374),
    "KITTI-17": (1242, 374),
    "PETS09-S2L1": (768, 576),
    "TUD-Campus": (640, 480),
    "TUD-Stadtmitte": (640, 480),
    "Venice-2": (1920, 1080)
    # Add more entries as needed...
}

MOT17_RESOLUTION_DICT = {
    "MOT17-02-DPM": (1920, 1080),
    "MOT17-02-FRCNN": (1920, 1080),
    "MOT17-02-SDP": (1920, 1080),
    "MOT17-05-DPM": (640, 480),
    "MOT17-05-FRCNN": (640, 480),
    "MOT17-05-SDP": (640, 480),
    "MOT17-04-DPM": (1920, 1080),
    "MOT17-04-FRCNN": (1920, 1080),
    "MOT17-04-SDP": (1920, 1080),
    "MOT17-09-DPM": (1920, 1080),
    "MOT17-09-FRCNN": (1920, 1080),
    "MOT17-09-SDP": (1920, 1080),
    "MOT17-10-DPM": (1920, 1080),
    "MOT17-10-FRCNN": (1920, 1080),
    "MOT17-10-SDP": (1920, 1080),
    "MOT17-11-DPM": (1920, 1080),
    "MOT17-11-FRCNN": (1920, 1080),
    "MOT17-11-SDP": (1920, 1080),
    "MOT17-13-DPM": (1920, 1080),
    "MOT17-13-FRCNN": (1920, 1080),
    "MOT17-13-SDP": (1920, 1080),
    # Add more entries as needed...
}

MOT16_RESOLUTION_DICT = {
    # Add more entries as needed...
}

MOT20_RESOLUTION_DICT = {
    # Add more entries as needed...
}


class JSONLogParser:
    def __init__(self, input_path: Path, output_path: Path,model_width: int, model_height: int):
        """
        Initializes the JSONLogParser with paths for input and output.

        Args:
            input_path (Path): Path to the input .log file.
            output_path (Path): Path to the output .txt file.
        """
        self.input_path = input_path
        self.output_path = output_path
        self.total_frames = 0
        self.total_objects = 0
        
        self.model_width = model_width
        self.model_height = model_height
        
        # Derive base filename to look up resolution
        base_name = input_path.stem.replace("-raw", "").replace(".mp4", "")
        # User can change to MOT17_RESOLUTION_DICT if log results are MOT17
        self.width, self.height = MOT17_RESOLUTION_DICT.get(base_name, (1920, 1080))  # Default to 320x320 if not found
        self.width_ratio = self.width / self.model_width
        self.height_ratio = self.height / self.model_height

    def parse_and_save_frame_order(self):
        """
        Parses the input JSON log file frame by frame and writes the extracted object
        data in the order of frame IDs to the output file.

        Input:
            - Reads from self.input_path
        Output:
            - Writes parsed data line-by-line to self.output_path
            - Logs object information per frame
        """
        with self.input_path.open('r') as infile, self.output_path.open('w') as outfile:
            for line_num, line in enumerate(infile, start=1):
                self._process_line(line.strip(), line_num, outfile)
        self._print_summary()
        
    def _process_line(self, line, line_num, outfile):
        """
        Helper function to parse a single line from the log file and write all object
        data (with bounding boxes) to the output file.

        Args:
            line (str): JSON string for one line of input.
            line_num (int): Line number in the input file, used for logging.
            outfile (TextIOWrapper): Opened output file handle to write data.
        """
        try:
            data = json.loads(line)
            frame_dict = data.get("frame_ID", {})
            cnt = 1
            ratio_x = self.width_ratio
            ratio_y = self.height_ratio
            for frame_str, frame_data in frame_dict.items():
                frame = int(frame_str)
                self.total_frames += 1
                track_objs = frame_data.get("trackObj", [])

                logging.info(f"📸 Frame {frame} ➜ {len(track_objs)} objects")
                logging.info(f"--------------------------------------")
                for obj in track_objs:
                    obj_id = int(obj.get("trackObj.id", -1))
                    x1 = obj.get("trackObj.x1", 0)
                    y1 = obj.get("trackObj.y1", 0)
                    x2 = obj.get("trackObj.x2", 0)
                    y2 = obj.get("trackObj.y2", 0)
                    conf = obj.get("trackObj.confidence", 0.0)

                    bb_left = int(x1 * ratio_x)
                    bb_top = int(y1 * ratio_y)
                    bb_width = int((x2 - x1) * ratio_x)
                    bb_height = int((y2 - y1) * ratio_y)

                    line_str = f"{frame},{obj_id},{bb_left},{bb_top},{bb_width},{bb_height},{conf:.6f},-1,-1,-1" # {conf:.6f}
                    logging.info(f"Track {cnt}: {line_str}")
                    outfile.write(line_str + "\n")

                    cnt += 1
                    self.total_objects += 1

        except json.JSONDecodeError as e:
            logging.warning(f"⚠️ Skipping invalid JSON at line {line_num}: {e}")


    def parse_and_save_obj_id_order(self):
        """
        Parses the input JSON log file and collects all objects across all frames,
        then sorts them by object ID and frame number before writing to the output file.

        Input:
            - Reads from self.input_path
        Output:
            - Writes sorted object entries by obj_id to self.output_path
            - Logs each object's bounding box and confidence
        """
        all_objects = []
        ratio = 1

        with self.input_path.open('r') as infile:
            for line_num, line in enumerate(infile, start=1):
                try:
                    data = json.loads(line.strip())
                    frame_dict = data.get("frame_ID", {})

                    for frame_str, frame_data in frame_dict.items():
                        frame = int(frame_str)
                        self.total_frames += 1
                        track_objs = frame_data.get("trackObj", [])

                        for obj in track_objs:
                            obj_id = obj.get("trackObj.id", -1)
                            x1 = obj.get("trackObj.x1", 0)
                            y1 = obj.get("trackObj.y1", 0)
                            x2 = obj.get("trackObj.x2", 0)
                            y2 = obj.get("trackObj.y2", 0)
                            conf = obj.get("trackObj.confidence", 0.0)

                            bb_left = int(x1) * ratio
                            bb_top = int(y1) * ratio
                            bb_width = int((x2 * ratio) - (x1 * ratio))
                            bb_height = int((y2 * ratio) - (y1 * ratio))

                            all_objects.append((obj_id, frame, bb_left, bb_top, bb_width, bb_height, conf))

                except json.JSONDecodeError as e:
                    logging.warning(f"⚠️ Skipping invalid JSON at line {line_num}: {e}")

        # Sort all objects by obj_id (ascending), then frame (optional secondary sort)
        all_objects.sort(key=lambda x: (x[0], x[1]))

        with self.output_path.open('w') as outfile:
            for cnt, (obj_id, frame, bb_left, bb_top, bb_width, bb_height, conf) in enumerate(all_objects, 1):
                line_str = f"{frame},{obj_id},{bb_left},{bb_top},{bb_width},{bb_height},{conf:.6f},-1,-1,-1"#{conf:.6f}
                logging.info(f"Track {cnt}: {line_str}")
                outfile.write(line_str + "\n")
                self.total_objects += 1

        self._print_summary()


    
    def _print_summary(self):
        """
        Logs a summary of total frames and objects processed.
        """
        logging.info("✅ Conversion completed!")
        logging.info(f"🧾 Total frames processed: {self.total_frames}")
        logging.info(f"👁️ Total objects saved: {self.total_objects}")

    @staticmethod
    def parse_all_logs_in_directory(input_dir: Path, output_dir: Path, model_width: int, model_height:int):
        """
        Parses all .log files in a directory and saves corresponding .txt files
        in the output directory using frame order parsing.

        Args:
            input_dir (Path): Directory containing .log files.
            output_dir (Path): Directory to store the converted .txt files.

        Note:
            - Only `parse_and_save_frame_order()` is currently used.
            - To sort by object ID, uncomment `parse_and_save_obj_id_order()`.
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        log_files = list(input_dir.glob("*.log"))

        if not log_files:
            logging.info("📭 No .log files found in the directory.")
            return

        for log_file in log_files:
            # Remove .log and .mp4, then remove '-raw' if present
            base_name = Path(log_file.stem).stem  # Removes .log then .mp4
            clean_name = base_name.replace("-raw", "")  # Removes "-raw"
            output_file = output_dir / f"{clean_name}.txt"

            logging.info(f"\n📂 Processing: {log_file.name}")
            parser = JSONLogParser(log_file, output_file,model_width,model_height)
            parser.parse_and_save_frame_order()
            # parser.parse_and_save_obj_id_order()
            
if __name__ == "__main__":
    input_directory = Path("../build/debug/logs")
    output_directory = Path("../build/debug/MOT_logs")  # You can change this if needed
    model_width = 512
    model_height = 288
    JSONLogParser.parse_all_logs_in_directory(input_directory, 
                                              output_directory,
                                              model_width,
                                              model_height)

