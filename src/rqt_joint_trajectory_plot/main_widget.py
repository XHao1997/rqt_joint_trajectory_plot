#!/usr/bin/env python3

from python_qt_binding import loadUi
from python_qt_binding.QtCore import Qt, QTimer, Signal, Slot
from python_qt_binding.QtGui import QIcon
from python_qt_binding.QtWidgets import QWidget, QTreeWidgetItem
import rospy
import rospkg
from roslib.message import get_message_class
from trajectory_msgs.msg import JointTrajectory
from control_msgs.msg import FollowJointTrajectoryActionGoal
from moveit_msgs.msg import DisplayTrajectory
from .plot_widget import PlotWidget
import numpy as np
import threading

class MainWidget(QWidget):
    draw_curves = Signal(list, dict)
    topics_refreshed = Signal()

    def __init__(self):
        super(MainWidget, self).__init__()
        self.setObjectName('MainWidget')

        # Load UI
        self._load_ui()
        
        # Initialize variables
        self.handler = None
        self.joint_names = []
        self.topic_name_class_map = {}
        self.time = None
        self.dis, self.vel = {}, {}

        # Initialize timer
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update)

        # Initialize plot widget
        self.plot_widget = PlotWidget(self)
        self.plot_layout.addWidget(self.plot_widget)
        self.draw_curves.connect(self.plot_widget.draw_curves)

        # Connect signals and slots
        self.refresh_button.clicked.connect(self.refresh_topics_in_thread)
        self.topic_combox.currentIndexChanged.connect(self.change_topic)
        self.select_tree.itemChanged.connect(self.update_checkbox)

        # Refresh topics initially
        self.refresh_topics_in_thread()
        self.change_topic()

    def _load_ui(self):
        """Load the UI file."""
        rospack = rospkg.RosPack()
        ui_file = rospack.get_path('rqt_joint_trajectory_plot') + '/resource/JointTrajectoryPlot.ui'
        loadUi(ui_file, self)

        # Set icons for buttons
        self.refresh_button.setIcon(QIcon.fromTheme('view-refresh'))
        self.pause_button.setIcon(QIcon.fromTheme('media-playback-pause'))

    def refresh_topics(self):
        """Refresh topic list in the combobox."""
        try:
            topic_list = rospy.get_published_topics()
        except rospy.ROSException as e:
            rospy.logwarn(f"Error fetching topics: {e}")
            return

        if not topic_list:
            rospy.loginfo("No topics found.")
            return

        self.topic_combox.clear()
        self.topic_name_class_map.clear()
        for name, type in topic_list:
            if type in [
                'trajectory_msgs/JointTrajectory',
                'control_msgs/FollowJointTrajectoryActionGoal',
                'moveit_msgs/DisplayTrajectory'
            ]:
                self.topic_name_class_map[name] = get_message_class(type)
                self.topic_combox.addItem(name)

        # Emit signal to indicate topics are refreshed
        self.topics_refreshed.emit()

    def refresh_topics_in_thread(self):
        """Run `refresh_topics` in a separate thread."""
        thread = threading.Thread(target=self.refresh_topics)
        thread.daemon = True
        thread.start()

    def change_topic(self):
        """Handle topic change."""
        topic_name = self.topic_combox.currentText()
        if not topic_name:
            return
        if self.handler:
            self.handler.unregister()

        self.joint_names = []
        self.handler = rospy.Subscriber(
            topic_name, rospy.AnyMsg, self.callback, topic_name, queue_size=200)

    def close(self):
        """Handle widget close event."""
        if self.handler:
            self.handler.unregister()
            self.handler = None

    def refresh_tree(self):
        """Refresh the tree widget with joint names."""
        self.select_tree.clear()
        for joint_name in self.joint_names:
            joint_item = QTreeWidgetItem(self.select_tree)
            joint_item.setText(0, joint_name)
            joint_item.setCheckState(0, Qt.Unchecked)
            joint_item.setFlags(Qt.ItemIsUserCheckable | Qt.ItemIsEnabled)
            for traj_name in ['position', 'velocity']:
                sub_item = QTreeWidgetItem(joint_item)
                sub_item.setText(0, traj_name)
                sub_item.setCheckState(0, Qt.Unchecked)
                sub_item.setFlags(Qt.ItemIsUserCheckable | Qt.ItemIsEnabled)

    def callback(self, anymsg, topic_name):
        """Callback to process incoming messages."""
        if self.pause_button.isChecked():
            return

        msg_class = self.topic_name_class_map.get(topic_name)
        if not msg_class:
            rospy.logerr(f"Unknown topic class for {topic_name}")
            return

        msg = self._deserialize_message(anymsg, msg_class)
        if not msg:
            return

        self._process_message(msg)
        self.plot_graph()

    def _deserialize_message(self, anymsg, msg_class):
        """Deserialize AnyMsg to a specific message type."""
        try:
            if msg_class == JointTrajectory:
                return JointTrajectory().deserialize(anymsg._buff)
            elif msg_class == FollowJointTrajectoryActionGoal:
                return FollowJointTrajectoryActionGoal().deserialize(anymsg._buff).goal.trajectory
        except Exception as e:
            rospy.logerr(f"Error deserializing message: {e}")
        return None

    def _process_message(self, msg):
        """Process a deserialized message."""
        self.time = np.array([point.time_from_start.to_sec() for point in msg.points])
        self.dis, self.vel = {}, {}

        for i, point in enumerate(msg.points):
            for j, joint_name in enumerate(msg.joint_names):
                if joint_name not in self.dis:
                    self.dis[joint_name] = np.zeros(len(msg.points))
                if joint_name not in self.vel:
                    self.vel[joint_name] = np.zeros(len(msg.points))

                if point.positions and j < len(point.positions):
                    self.dis[joint_name][i] = point.positions[j]
                if point.velocities and j < len(point.velocities):
                    self.vel[joint_name][i] = point.velocities[j]

        if self.joint_names != msg.joint_names:
            self.joint_names = msg.joint_names
            self.refresh_tree()


    def plot_graph(self):
        """Emit signal to update the plot."""
        curve_names = []
        data = {}
        traj_names = ['position', 'velocity']
        data_list = [self.dis, self.vel]

        for i in range(self.select_tree.topLevelItemCount()):
            joint_item = self.select_tree.topLevelItem(i)
            for n, traj_name in enumerate(traj_names):
                item = joint_item.child(n)
                if item.checkState(0) == Qt.Checked:
                    joint_name = joint_item.text(0)
                    curve_name = f"{self.label_reference.text()}/{joint_name} {traj_name}"
                    curve_names.append(curve_name)
                    data[curve_name] = (self.time, data_list[n][joint_name])
        self.draw_curves.emit(curve_names, data)

    def update_checkbox(self, item, column):
        """Update checkboxes recursively and refresh plot."""
        self._recursive_check(item)
        self.plot_graph()

    def _recursive_check(self, item):
        """Recursively check or uncheck items."""
        check_state = item.checkState(0)
        for i in range(item.childCount()):
            child = item.child(i)
            child.setCheckState(0, check_state)
            self._recursive_check(child)
