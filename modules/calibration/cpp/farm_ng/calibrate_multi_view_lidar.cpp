#include <iostream>
#include <optional>
#include <sstream>
#include <stdexcept>

#include <gflags/gflags.h>
#include <glog/logging.h>
#include <boost/asio.hpp>

#include "farm_ng/calibration/calibrate_multi_view_apriltag_rig.pb.h"
#include "farm_ng/calibration/calibrate_multi_view_lidar.pb.h"

#include "farm_ng/core/blobstore.h"
#include "farm_ng/core/init.h"
#include "farm_ng/core/ipc.h"

DEFINE_bool(interactive, false, "receive program args via eventbus");
DEFINE_string(output_config, "", "Output the config to a file.");
DEFINE_string(config, "", "Load config from a file rather than args.");

DEFINE_string(name, "", "Name of calibration output.");
DEFINE_string(event_log, "", "Path to event log containing input data.");
DEFINE_string(
    calibrate_multi_view_apriltag_rig_result, "",
    "Path to result of calibrate_multi_view_apriltag_rig, containing camera "
    "rig and apriltag rig to calibrate the LIDARs with respect to.");

namespace fs = boost::filesystem;

namespace farm_ng::calibration {

class CalibrateMultiViewLidarProgram {
 public:
  CalibrateMultiViewLidarProgram(
      core::EventBus& bus, CalibrateMultiViewLidarConfiguration configuration,
      bool interactive)
      : bus_(bus), timer_(bus_.get_io_service()) {
    if (interactive) {
      status_.mutable_input_required_configuration()->CopyFrom(configuration);
    } else {
      set_configuration(configuration);
    }
    bus_.AddSubscriptions({"^" + bus_.GetName() + "/"});
    bus_.GetEventSignal()->connect(
        std::bind(&CalibrateMultiViewLidarProgram::on_event, this,
                  std::placeholders::_1));
    on_timer(boost::system::error_code());
  }

  int run() {
    while (status_.has_input_required_configuration()) {
      bus_.get_io_service().run_one();
    }
    LOG(INFO) << "config:\n" << configuration_.DebugString();

    CalibrateMultiViewLidarResult result;

    send_status();
    return 0;
  }

  void send_status() {
    bus_.Send(core::MakeEvent(bus_.GetName() + "/status", status_));
  }

  void on_timer(const boost::system::error_code& error) {
    if (error) {
      LOG(WARNING) << "timer error: " << __PRETTY_FUNCTION__ << error;
      return;
    }
    timer_.expires_from_now(boost::posix_time::millisec(1000));
    timer_.async_wait(std::bind(&CalibrateMultiViewLidarProgram::on_timer, this,
                                std::placeholders::_1));

    send_status();
  }

  bool on_configuration(const core::Event& event) {
    CalibrateMultiViewLidarConfiguration configuration;
    if (!event.data().UnpackTo(&configuration)) {
      return false;
    }
    LOG(INFO) << configuration.ShortDebugString();
    set_configuration(configuration);
    return true;
  }

  void set_configuration(CalibrateMultiViewLidarConfiguration configuration) {
    configuration_ = configuration;
    status_.clear_input_required_configuration();
    send_status();
  }

  void on_event(const core::Event& event) {
    CHECK(event.name().rfind(bus_.GetName() + "/", 0) == 0);
    if (on_configuration(event)) {
      return;
    }
  }

 private:
  core::EventBus& bus_;
  boost::asio::deadline_timer timer_;
  CalibrateMultiViewLidarConfiguration configuration_;
  CalibrateMultiViewLidarStatus status_;
  CalibrateMultiViewLidarResult result_;
};

}  // namespace farm_ng::calibration

void Cleanup(farm_ng::core::EventBus& bus) {}

int Main(farm_ng::core::EventBus& bus) {
  farm_ng::calibration::CalibrateMultiViewLidarConfiguration config;
  if (!FLAGS_config.empty()) {
    config = farm_ng::core::ReadProtobufFromJsonFile<
        farm_ng::calibration::CalibrateMultiViewLidarConfiguration>(
        FLAGS_config);
    farm_ng::calibration::CalibrateMultiViewLidarProgram program(
        bus, config, FLAGS_interactive);
    return program.run();
  } else {
    config.set_name(FLAGS_name);
    config.mutable_event_log()->CopyFrom(
        farm_ng::core::EventLogResource(FLAGS_event_log));
    config.mutable_calibrate_multi_view_apriltag_rig_result()->CopyFrom(
        farm_ng::core::ProtobufJsonResource<
            farm_ng::calibration::CalibrateMultiViewApriltagRigResult>(
            FLAGS_calibrate_multi_view_apriltag_rig_result));
  }
  if (!FLAGS_output_config.empty()) {
    farm_ng::core::WriteProtobufToJsonFile(FLAGS_output_config, config);
    return 0;
  }
  LOG(ERROR) << "Please provide a config.";
  return -1;
}
int main(int argc, char* argv[]) {
  return farm_ng::core::Main(argc, argv, &Main, &Cleanup);
}
