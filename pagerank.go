// Copyright 2025 The aedoom Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"fmt"
	"math/rand"
)

// PageRank is a pagerank based model
type PageRank struct {
	rng    *rand.Rand
	markov [][256][256]uint64
	w, h   int
	action TypeAction
}

// NewPageRank new pagerank model
func NewPageRank() *PageRank {
	return &PageRank{
		rng: rand.New(rand.NewSource(1)),
	}
}

// Process processes a frame
func (p *PageRank) Process(img Frame) (bool, TypeAction) {
	width := img.Frame.Bounds().Max.X
	height := img.Frame.Bounds().Max.Y
	if p.markov == nil {
		w, h := width/4, height/4
		fmt.Println(width, height, w, h, w*h)
		p.markov = make([][256][256]uint64, w*h+int(ActionCount))
		p.w, p.h = w, h
	}
	index, previous := 0, img.GrayAt(0, 0).Y
	for y := 0; y < height-8; y += 8 {
		for x := 0; x < width-8; x += 8 {
			for yy := 0; yy < 8; yy++ {
				for xx := 0; xx < 8; xx++ {
					current := img.GrayAt(x+xx, y+yy).Y
					p.markov[index][previous][current]++
					previous = current
				}
			}
			index++
		}
	}
	return false, ActionCount
}
