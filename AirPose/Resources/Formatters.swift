import Foundation

extension Date {
    var airPoseMediumString: String {
        formatted(date: .abbreviated, time: .shortened)
    }
}

extension Double {
    var airPoseOneDecimalString: String {
        formatted(.number.precision(.fractionLength(1)))
    }

    var airPoseCentimeterString: String {
        "\(airPoseOneDecimalString) cm"
    }

    var airPoseMillisecondsString: String {
        "\(airPoseOneDecimalString) ms"
    }

    var airPoseDegreeString: String {
        "\(airPoseOneDecimalString)°"
    }

    var airPoseScoreString: String {
        airPoseOneDecimalString
    }

    var airPoseRatioString: String {
        airPoseOneDecimalString
    }
}

extension Bool {
    var airPosePassFailString: String {
        self ? "Pass" : "Fail"
    }
}
